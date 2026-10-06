import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple
import numpy as np
import config
from tokenizer_utils import ScratchCodeTokenizer
from embedding_model import ScratchCodeEmbeddingModel, InfoNCELoss

class CodeTextPairDataset(Dataset):
    """PyTorch Dataset for Code-Text pairs."""
    def __init__(self, pairs: List[Dict[str, Any]], tokenizer: ScratchCodeTokenizer, max_len: int = config.MAX_SEQ_LEN):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        text_ids = self.tokenizer.encode(item["text"], max_length=self.max_len)
        code_ids = self.tokenizer.encode(item["code"], max_length=self.max_len)
        return {
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "code_ids": torch.tensor(code_ids, dtype=torch.long),
            "language": item.get("language", "unknown"),
            "file_path": item.get("file_path", "unknown")
        }

def evaluate_metrics(
    model: ScratchCodeEmbeddingModel,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Evaluates Recall@1, Recall@5, and MRR (Mean Reciprocal Rank) overall and per language.
    """
    model.eval()
    all_text_embeds = []
    all_code_embeds = []
    all_langs = []

    with torch.no_grad():
        for batch in test_loader:
            text_ids = batch["text_ids"].to(device)
            code_ids = batch["code_ids"].to(device)
            
            t_emb = model(text_ids)
            c_emb = model(code_ids)
            
            all_text_embeds.append(t_emb.cpu().numpy())
            all_code_embeds.append(c_emb.cpu().numpy())
            all_langs.extend(batch["language"])

    if not all_text_embeds:
        return {"recall@1": 0.0, "recall@5": 0.0, "mrr": 0.0}, {}

    Q = np.concatenate(all_text_embeds, axis=0)  # (N, D)
    C = np.concatenate(all_code_embeds, axis=0)  # (N, D)
    N = Q.shape[0]

    if N == 0:
        return {"recall@1": 0.0, "recall@5": 0.0, "mrr": 0.0}, {}

    # Compute Similarity Matrix (N, N)
    sim_matrix = np.dot(Q, C.T)

    recalls_1 = []
    recalls_5 = []
    mrrs = []

    lang_metrics_raw = {}

    for i in range(N):
        scores = sim_matrix[i]
        # Sort indices in descending order
        sorted_indices = np.argsort(-scores)
        
        # Rank of correct item i (1-indexed)
        rank = int(np.where(sorted_indices == i)[0][0]) + 1

        r1 = 1.0 if rank == 1 else 0.0
        r5 = 1.0 if rank <= 5 else 0.0
        mrr = 1.0 / rank

        recalls_1.append(r1)
        recalls_5.append(r5)
        mrrs.append(mrr)

        lang = all_langs[i]
        if lang not in lang_metrics_raw:
            lang_metrics_raw[lang] = {"r1": [], "r5": [], "mrr": []}
        lang_metrics_raw[lang]["r1"].append(r1)
        lang_metrics_raw[lang]["r5"].append(r5)
        lang_metrics_raw[lang]["mrr"].append(mrr)

    overall_metrics = {
        "recall@1": float(np.mean(recalls_1)),
        "recall@5": float(np.mean(recalls_5)),
        "mrr": float(np.mean(mrrs))
    }

    per_language_metrics = {}
    for lang, data in lang_metrics_raw.items():
        per_language_metrics[lang] = {
            "recall@1": float(np.mean(data["r1"])),
            "recall@5": float(np.mean(data["r5"])),
            "mrr": float(np.mean(data["mrr"])),
            "sample_count": len(data["r1"])
        }

    return overall_metrics, per_language_metrics

class ScratchModelTrainer:
    """
    Trains custom Scratch Code Embedding Model using InfoNCE Loss with Early Stopping.
    """
    def __init__(self, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Trainer using device: {self.device}")

    def train_and_evaluate(
        self,
        train_pairs: List[Dict[str, Any]],
        val_pairs: List[Dict[str, Any]],
        test_pairs: List[Dict[str, Any]],
        epochs: int = config.NUM_EPOCHS,
        batch_size: int = config.BATCH_SIZE,
        lr: float = config.LEARNING_RATE
    ) -> Tuple[ScratchCodeEmbeddingModel, ScratchCodeTokenizer, Dict[str, Any]]:
        """
        Trains tokenizer & model from scratch, evaluates held-out test split, and returns trained weights.
        """
        # 1. Train Scratch Tokenizer on all texts & code
        all_corpus = [p["text"] for p in train_pairs + val_pairs] + [p["code"] for p in train_pairs + val_pairs]
        tokenizer = ScratchCodeTokenizer(vocab_size=config.VOCAB_SIZE)
        tokenizer.train_from_texts(all_corpus, save_path=config.TOKENIZER_PATH)

        # 2. Build Datasets & DataLoaders
        train_ds = CodeTextPairDataset(train_pairs, tokenizer)
        val_ds = CodeTextPairDataset(val_pairs, tokenizer)
        test_ds = CodeTextPairDataset(test_pairs, tokenizer)

        # Adjust batch size if dataset is smaller
        eff_batch_size = min(batch_size, max(2, len(train_ds)))

        train_loader = DataLoader(train_ds, batch_size=eff_batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=max(2, min(eff_batch_size, len(val_ds))), shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=max(1, min(eff_batch_size, len(test_ds))), shuffle=False)

        # 3. Model & Optimizer Setup
        model = ScratchCodeEmbeddingModel(
            vocab_size=config.VOCAB_SIZE,
            embed_dim=config.EMBEDDING_DIM,
            proj_dim=config.PROJECTION_DIM,
            max_seq_len=config.MAX_SEQ_LEN
        ).to(self.device)

        criterion = InfoNCELoss(temperature=config.INFO_NCE_TEMPERATURE).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        best_val_loss = float("inf")
        patience = 3
        patience_counter = 0

        print("\n--- Starting Training (100% Scratch Weights, InfoNCE Loss) ---")
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                text_ids = batch["text_ids"].to(self.device)
                code_ids = batch["code_ids"].to(self.device)

                optimizer.zero_grad()
                q_embeds = model(text_ids)
                c_embeds = model(code_ids)

                loss = criterion(q_embeds, c_embeds)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= max(1, len(train_loader))

            # Validation step
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    text_ids = batch["text_ids"].to(self.device)
                    code_ids = batch["code_ids"].to(self.device)

                    q_embeds = model(text_ids)
                    c_embeds = model(code_ids)

                    loss = criterion(q_embeds, c_embeds)
                    val_loss += loss.item()

            val_loss /= max(1, len(val_loader))
            print(f"Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), config.MODEL_WEIGHTS_PATH)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break

        # Load best weights for evaluation
        if os.path.exists(config.MODEL_WEIGHTS_PATH):
            model.load_state_dict(torch.load(config.MODEL_WEIGHTS_PATH, map_location=self.device))

        # Evaluate on Held-out Test Split
        overall_metrics, per_lang_metrics = evaluate_metrics(model, test_loader, self.device)
        
        results = {
            "overall": overall_metrics,
            "per_language": per_lang_metrics,
            "train_samples": len(train_pairs),
            "val_samples": len(val_pairs),
            "test_samples": len(test_pairs)
        }

        print("\n================ Held-out Test Set Evaluation ================")
        print(f"Overall Recall@1: {overall_metrics['recall@1']*100:.2f}%")
        print(f"Overall Recall@5: {overall_metrics['recall@5']*100:.2f}%")
        print(f"Overall MRR:      {overall_metrics['mrr']:.4f}")
        print("---------------- Per-Language Breakdown ----------------")
        for lang, m in per_lang_metrics.items():
            print(f" [{lang}] Count: {m['sample_count']} | R@1: {m['recall@1']*100:.2f}% | R@5: {m['recall@5']*100:.2f}% | MRR: {m['mrr']:.4f}")
        print("=============================================================\n")

        return model, tokenizer, results

import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
import config
from tokenizer_utils import ScratchCodeTokenizer
from embedding_model import ScratchCodeEmbeddingModel

class VectorStore:
    """
    Local Vector Store using NumPy (.npz) for embeddings and JSON (.json) for metadata.
    Provides indexing and cosine similarity search over repository chunks.
    """
    def __init__(
        self,
        npz_path: str = config.VECTOR_STORE_NPZ,
        json_path: str = config.VECTOR_STORE_JSON
    ):
        self.npz_path = npz_path
        self.json_path = json_path
        self.vectors: np.ndarray = None  # (N, D)
        self.metadata: List[Dict[str, Any]] = []

    def index(
        self,
        chunks: List[Dict[str, Any]],
        model: ScratchCodeEmbeddingModel,
        tokenizer: ScratchCodeTokenizer,
        batch_size: int = 32,
        device: torch.device = None
    ) -> int:
        """
        Embeds all code chunks and persists vectors and metadata to disk.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        model.eval()

        self.metadata = chunks
        chunk_texts = [c["text"] for c in chunks]

        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[i:i + batch_size]
                batch_ids = [tokenizer.encode(t) for t in batch_texts]
                input_tensor = torch.tensor(batch_ids, dtype=torch.long, device=device)
                
                embeddings = model(input_tensor)
                all_embeddings.append(embeddings.cpu().numpy())

        if all_embeddings:
            self.vectors = np.concatenate(all_embeddings, axis=0)
        else:
            self.vectors = np.empty((0, config.PROJECTION_DIM), dtype=np.float32)

        # Save to disk
        os.makedirs(os.path.dirname(self.npz_path), exist_ok=True)
        np.savez_compressed(self.npz_path, vectors=self.vectors)

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"Indexed {len(chunks)} code chunks -> Vectors: {self.npz_path}, Metadata: {self.json_path}")
        return len(chunks)

    def load(self) -> bool:
        """Loads index vectors and metadata from disk."""
        if os.path.exists(self.npz_path) and os.path.exists(self.json_path):
            data = np.load(self.npz_path)
            self.vectors = data["vectors"]
            with open(self.json_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            return True
        return False

    def search(
        self,
        question: str,
        model: ScratchCodeEmbeddingModel,
        tokenizer: ScratchCodeTokenizer,
        top_k: int = 5,
        device: torch.device = None
    ) -> List[Dict[str, Any]]:
        """
        Computes Cosine Similarity between question vector and chunk vectors.
        Returns top-k matching chunks with similarity score.
        """
        if self.vectors is None or len(self.vectors) == 0:
            if not self.load():
                return []

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.to(device)
        model.eval()

        # Encode question
        q_ids = tokenizer.encode(question)
        input_tensor = torch.tensor([q_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            q_emb = model(input_tensor).cpu().numpy()[0]  # (D,)

        # Vectors and q_emb are already L2 normalized, so dot product = Cosine Similarity!
        similarities = np.dot(self.vectors, q_emb)  # (N,)

        # Sort top-k
        top_indices = np.argsort(-similarities)[:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            chunk = self.metadata[idx].copy()
            chunk["score"] = score
            results.append(chunk)

        return results

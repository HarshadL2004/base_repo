import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class ScratchCodeEmbeddingModel(nn.Module):
    """
    100% Custom Dual-Encoder Embedding Architecture trained strictly from scratch.
    Weights are randomly initialized using Xavier/Kaiming uniform distributions.
    """
    def __init__(
        self,
        vocab_size: int = config.VOCAB_SIZE,
        embed_dim: int = config.EMBEDDING_DIM,
        proj_dim: int = config.PROJECTION_DIM,
        max_seq_len: int = config.MAX_SEQ_LEN,
        num_layers: int = 2,
        nhead: int = 4,
        pad_idx: int = 0
    ):
        super(ScratchCodeEmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx

        # Token & Positional Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)

        # Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Linear Projection & Normalization Layer
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

        self._reset_weights()

    def _reset_weights(self):
        """Randomly initialize all network weights from scratch."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].fill_(0.0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: Tensor of shape (batch_size, seq_len)
        Returns: L2-normalized embeddings of shape (batch_size, proj_dim)
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # Ensure sequence length does not exceed max_seq_len
        if seq_len > self.max_seq_len:
            input_ids = input_ids[:, :self.max_seq_len]
            seq_len = self.max_seq_len

        # Positional indices
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

        # Padding mask: True where padded
        key_padding_mask = (input_ids == self.pad_idx)

        # Sum Token + Positional embeddings
        x = self.token_embeddings(input_ids) + self.position_embeddings(positions)

        # Pass through Transformer Encoder
        out = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        # Masked Mean Pooling over non-padding tokens
        mask = (~key_padding_mask).unsqueeze(-1).float()  # (batch, seq, 1)
        sum_embeddings = torch.sum(out * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask

        # Projection head
        projected = self.proj_head(pooled)

        # L2 Normalization
        normalized_embeddings = F.normalize(projected, p=2, dim=1)
        return normalized_embeddings


class InfoNCELoss(nn.Module):
    """
    Contrastive InfoNCE Loss function for training text-code pairs.
    """
    def __init__(self, temperature: float = config.INFO_NCE_TEMPERATURE):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, query_embeddings: torch.Tensor, code_embeddings: torch.Tensor) -> torch.Tensor:
        """
        query_embeddings: (batch_size, proj_dim)
        code_embeddings: (batch_size, proj_dim)
        """
        device = query_embeddings.device
        batch_size = query_embeddings.size(0)

        # Compute Cosine Similarity Matrix scaled by temperature
        sim_matrix = torch.matmul(query_embeddings, code_embeddings.T) / self.temperature

        labels = torch.arange(batch_size, device=device)

        # Symmetric Loss (Query-to-Code + Code-to-Query)
        loss_q2c = self.cross_entropy(sim_matrix, labels)
        loss_c2q = self.cross_entropy(sim_matrix.T, labels)

        return (loss_q2c + loss_c2q) / 2.0

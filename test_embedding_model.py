import torch
from tokenizer_utils import ScratchCodeTokenizer
from embedding_model import ScratchCodeEmbeddingModel, InfoNCELoss

def test_tokenizer_and_embedding():
    print("--- Testing Phase 3: Scratch Tokenizer & InfoNCE Embedding Model ---")

    # Sample corpus
    corpus = [
        "def calculate_area(width, height): return width * height",
        "function computeArea(w, h) { return w * h; }",
        "func AddNumbers(a int, b int) int { return a + b }",
        "class Rectangle { float area() { return w * h; } }"
    ]

    # Train tokenizer from scratch
    tokenizer = ScratchCodeTokenizer(vocab_size=100)
    tokenizer.train_from_texts(corpus)

    # Encode sample
    sample_text = "def calculate_area(w, h): pass"
    encoded_ids = tokenizer.encode(sample_text, max_length=16)
    print(f"Sample Text: '{sample_text}'")
    print(f"Encoded Token IDs (length {len(encoded_ids)}): {encoded_ids}")

    assert len(encoded_ids) == 16, "Expected padded length 16"

    # Instantiate model
    model = ScratchCodeEmbeddingModel(vocab_size=100, embed_dim=32, proj_dim=16, max_seq_len=16)
    input_tensor = torch.tensor([encoded_ids, encoded_ids], dtype=torch.long)

    # Forward pass
    output_embeddings = model(input_tensor)
    print(f"Output Embedding Tensor Shape: {output_embeddings.shape}")

    # Check L2 norm
    norms = torch.norm(output_embeddings, p=2, dim=1)
    print(f"L2 Norms of output vectors: {norms.tolist()}")
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "Embeddings must be L2 normalized!"

    # InfoNCE loss test
    loss_fn = InfoNCELoss(temperature=0.07)
    q_embeds = torch.randn(4, 16)
    c_embeds = torch.randn(4, 16)
    q_norm = torch.nn.functional.normalize(q_embeds, p=2, dim=1)
    c_norm = torch.nn.functional.normalize(c_embeds, p=2, dim=1)

    loss = loss_fn(q_norm, c_norm)
    print(f"InfoNCE Loss computation test output: {loss.item():.4f}")
    assert loss.item() > 0.0, "Loss must be positive!"

    print("Phase 3 Scratch Tokenizer & Embedding Model Test PASSED successfully!")

if __name__ == "__main__":
    test_tokenizer_and_embedding()

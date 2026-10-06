import torch
from tokenizer_utils import ScratchCodeTokenizer
from embedding_model import ScratchCodeEmbeddingModel
from vector_store import VectorStore

def test_vector_store():
    print("--- Testing Phase 5: Vector Store ---")

    # Sample chunks
    chunks = [
        {
            "file_path": "math.py",
            "start_line": 1,
            "end_line": 3,
            "language": "python",
            "name": "multiply",
            "kind": "function",
            "params": ["a", "b"],
            "docstring": "Multiplies two numbers",
            "text": "def multiply(a, b):\n    return a * b"
        },
        {
            "file_path": "string_utils.js",
            "start_line": 1,
            "end_line": 4,
            "language": "javascript",
            "name": "capitalize",
            "kind": "function",
            "params": ["str"],
            "docstring": "Capitalizes first letter",
            "text": "function capitalize(str) {\n    return str.charAt(0).toUpperCase() + str.slice(1);\n}"
        }
    ]

    corpus = [c["text"] for c in chunks]
    tokenizer = ScratchCodeTokenizer(vocab_size=100)
    tokenizer.train_from_texts(corpus)

    model = ScratchCodeEmbeddingModel(vocab_size=100, embed_dim=32, proj_dim=16, max_seq_len=32)

    vstore = VectorStore()
    indexed_count = vstore.index(chunks, model, tokenizer)
    assert indexed_count == 2, f"Expected 2 indexed chunks, got {indexed_count}"

    # Search query
    results = vstore.search("how to multiply numbers", model, tokenizer, top_k=2)
    print(f"Search results for 'how to multiply numbers':")
    for r in results:
        print(f" - [{r['file_path']} L{r['start_line']}-{r['end_line']}] Score: {r['score']:.4f} Name: {r['name']}")

    assert len(results) == 2
    assert "score" in results[0]

    print("Phase 5 Vector Store Test PASSED successfully!")

if __name__ == "__main__":
    test_vector_store()

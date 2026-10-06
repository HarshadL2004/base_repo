import os
from dataset_builder import DatasetBuilder
from trainer import ScratchModelTrainer

def test_trainer():
    print("--- Testing Phase 4: Dataset Builder & Trainer ---")

    # Sample parsed chunks across 4 distinct files
    sample_chunks = [
        # File 1: Python
        {
            "file_path": "src/math_utils.py",
            "start_line": 1,
            "end_line": 5,
            "language": "python",
            "name": "add",
            "kind": "function",
            "params": ["a", "b"],
            "docstring": "Returns the sum of two numbers a and b.",
            "text": "def add(a, b):\n    return a + b"
        },
        {
            "file_path": "src/math_utils.py",
            "start_line": 7,
            "end_line": 10,
            "language": "python",
            "name": "multiply",
            "kind": "function",
            "params": ["x", "y"],
            "docstring": "Calculates product of x and y.",
            "text": "def multiply(x, y):\n    return x * y"
        },
        # File 2: JavaScript
        {
            "file_path": "src/api.js",
            "start_line": 1,
            "end_line": 4,
            "language": "javascript",
            "name": "fetchData",
            "kind": "function",
            "params": ["url"],
            "docstring": "Fetches raw data from given API endpoint url.",
            "text": "async function fetchData(url) {\n    return await fetch(url);\n}"
        },
        # File 3: Go
        {
            "file_path": "pkg/server.go",
            "start_line": 1,
            "end_line": 6,
            "language": "go",
            "name": "StartServer",
            "kind": "function",
            "params": ["port"],
            "docstring": "Starts HTTP server listening on specified port.",
            "text": "func StartServer(port int) error {\n    return http.ListenAndServe(port, nil)\n}"
        },
        # File 4: Java
        {
            "file_path": "com/app/User.java",
            "start_line": 1,
            "end_line": 8,
            "language": "java",
            "name": "getName",
            "kind": "function",
            "params": [],
            "docstring": "Gets user full name string.",
            "text": "public String getName() {\n    return this.name;\n}"
        }
    ]

    builder = DatasetBuilder()
    pairs = builder.build_repo_dataset_pairs(sample_chunks)
    print(f"Extracted {len(pairs)} training pairs from sample chunks.")

    train_p, val_p, test_p = builder.split_dataset_by_file(pairs, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    print(f"Split results: Train={len(train_p)}, Val={len(val_p)}, Test={len(test_p)}")

    # Verify no file leakage between train and test splits
    train_files = set(p["file_path"] for p in train_p)
    test_files = set(p["file_path"] for p in test_p)
    overlap = train_files.intersection(test_files)
    print(f"File leakage overlap between Train & Test: {overlap}")
    assert len(overlap) == 0, "No file should leak between Train and Test splits!"

    # Train model for 2 epochs on sample data
    trainer = ScratchModelTrainer()
    model, tokenizer, metrics = trainer.train_and_evaluate(
        train_pairs=train_p,
        val_pairs=val_p,
        test_pairs=test_p,
        epochs=2,
        batch_size=2
    )

    assert "overall" in metrics
    assert "recall@1" in metrics["overall"]
    print("Phase 4 Dataset Builder & Trainer Test PASSED successfully!")

if __name__ == "__main__":
    test_trainer()

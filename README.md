# 💻 Ask Your Codebase (100% Local Multi-Language AI Assistant)

**Ask Your Codebase** is a 100% local, privacy-first tool that takes a project **ZIP file** or **Git Repository URL**, ingests the entire codebase, trains a **custom PyTorch embedding model from scratch** (with zero pretrained weights or external APIs), stores embeddings in `.npz` + `.json`, and answers user questions using a local **Ollama** LLM (or graceful chunk fallback mode).

---

## 🌟 Key Features

1. **Loader & Filters**: Accepts `.zip` or Git repository URL (shallow clone `--depth 1`). Automatically skips `node_modules`, `.git`, binary files, images, and files > 1 MB.
2. **Multi-Language Chunker**: Structurally parses source code files into functions and classes across **Python, JavaScript/TypeScript, Java, Go, C/C++, PHP, Ruby**, and splits non-code files (Markdown, JSON, YAML) into paragraph blocks.
3. **100% Scratch PyTorch Embedding Model**:
   - Built with random initial weights (0% pretrained checkpoints loaded).
   - Code-aware pre-tokenization + custom BPE tokenizer trained from scratch using Hugging Face `tokenizers`.
   - PyTorch Transformer Encoder architecture with mean pooling and L2 normalization projection head.
   - Trained using **Contrastive InfoNCE Loss**.
4. **Project Adaptation & File-Based Split**:
   - Automatically extracts docstring/comment training pairs from ingested repos.
   - Performs an 80% Train / 10% Val / 10% Test split **by file** (preventing code leakage between train and test).
   - Early stopping on validation loss + held-out test evaluation reporting **Recall@1**, **Recall@5**, and **MRR**.
5. **NumPy & JSON Vector Store**: Saves L2-normalized embeddings in `chunk_vectors.npz` and metadata in `chunk_meta.json`. Computes top-k matches using Cosine Similarity.
6. **Local Answer Model (Ollama)**: Connects to local Ollama server at `http://localhost:11434`. Restricts answers strictly to provided code context with line numbers.
7. **Graceful Fallback**: If Ollama is not installed or offline, displays top matching code chunks with file paths and line ranges.
8. **Interactive Streamlit UI & 30-Question Benchmark**: Clean web interface + evaluation script for 30 multi-language sample questions.

---

## 🛠️ Installation & Setup Guide

### Step 1: Install Python Dependencies
Ensure Python 3.10+ is installed on your system. Open terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

### Step 2: Install & Setup Local Ollama Model (Recommended)
To enable local question answering:

1. **Download Ollama**: Visit [https://ollama.com](https://ollama.com) and install Ollama for Windows.
2. **Pull an Open-Weight Model**:
   Open PowerShell or Command Prompt and run:
   ```bash
   ollama pull qwen2.5-coder:1.5b
   ```
   *(Or for 8GB+ RAM: `ollama pull qwen2.5-coder:7b` or `ollama pull llama3.2:1b`)*
3. **Test Ollama Server**:
   Check that Ollama is running at `http://localhost:11434`:
   ```bash
   curl http://localhost:11434/api/tags
   ```

---

## 🚀 Running the Streamlit Web Application

Start the web application by running:

```bash
streamlit run app.py
```

1. Open your browser at `http://localhost:8501`.
2. Select **ZIP File Upload** or paste a **Git Repository URL**.
3. Click **🚀 Index & Train Codebase**.
4. Watch the step-by-step progress bars as the model trains from scratch and indexes vectors.
5. Review the **Held-Out Test Set Evaluation Metrics** (Recall@1, Recall@5, MRR).
6. Chat with your codebase in natural language!

---

## 🧪 Testing & Verification

Run individual component tests to verify every phase:

```bash
# Phase 1: Loader Test
python test_loader.py

# Phase 2: Chunker & Parser Test
python test_parser.py

# Phase 3: Scratch Tokenizer & InfoNCE Embedding Model Test
python test_embedding_model.py

# Phase 4: Dataset Builder & Trainer Test
python test_trainer.py

# Phase 5: Vector Store Test
python test_vector_store.py

# Phase 6: Answer Generator & Fallback Mode Test
python test_answer_model.py

# Phase 8: Complete 30-Question Benchmark Suite Test
python evaluate_test_script.py
```

---

## 📚 Real Public Code Datasets Download Guide

To pre-train or augment the embedding model with large public code datasets:

1. **CodeSearchNet** (Python, Java, JS, Go, PHP, Ruby):
   - GitHub: https://github.com/github/CodeSearchNet
   - Hugging Face: https://huggingface.co/datasets/code_search_net
   ```python
   from datasets import load_dataset
   ds = load_dataset("code_search_net", "python")
   ```

2. **CodeParrot Clean Python Dataset**:
   - Hugging Face: https://huggingface.co/datasets/codeparrot/codeparrot-clean
   ```python
   from datasets import load_dataset
   ds = load_dataset("codeparrot/codeparrot-clean", split="train")
   ```

3. **CodeXGLUE**:
   - GitHub: https://github.com/microsoft/CodeXGLUE

4. **The Stack**:
   - Hugging Face: https://huggingface.co/datasets/bigcode/the-stack

---

## 📊 System Architecture

```
[ZIP File / Git Repo URL]
        │
        ▼
   [loader.py] ── Filter node_modules, .git, binaries (>1MB)
        │
        ▼
   [parser.py] ── Multi-Language Function/Class AST Chunker
        │
        ▼
[dataset_builder.py] ── Extract docstrings + Repo Adaptation Pairs
        │
        ▼ (Split 80% Train / 10% Val / 10% Test BY FILE)
   [trainer.py] ── Train Tokenizer + PyTorch Model (InfoNCE Loss)
        │
        ▼
[vector_store.py] ── Save Embeddings (.npz) & Metadata (.json)
        │
        ▼
[answer_model.py] ── Query Ollama (http://localhost:11434) or Fallback
        │
        ▼
    [app.py] ── Streamlit UI
```

import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
TEMP_DIR = os.path.join(BASE_DIR, "temp_repos")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Loader Settings
MAX_FILE_SIZE_BYTES = 1024 * 1024  # 1 MB Limit
IGNORED_DIRS = {
    "node_modules", ".git", ".venv", "__pycache__", "build", "dist",
    ".idea", ".vscode", "target", "bin", "obj", ".next", ".pytest_cache"
}
IGNORED_EXTENSIONS = {
    ".zip", ".tar", ".gz", ".7z", ".rar", ".png", ".jpg", ".jpeg", ".gif",
    ".svg", ".ico", ".pdf", ".exe", ".so", ".dll", ".pyc", ".wasm", ".db",
    ".sqlite", ".bin", ".dat", ".pkl", ".pth", ".pt", ".onnx", ".woff", ".woff2"
}

# Embedding & Tokenizer Settings
VOCAB_SIZE = 8000
MAX_SEQ_LEN = 256
EMBEDDING_DIM = 128
PROJECTION_DIM = 128
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 10
INFO_NCE_TEMPERATURE = 0.07
RANDOM_SEED = 42

MODEL_WEIGHTS_PATH = os.path.join(ARTIFACTS_DIR, "scratch_embedding_model.pt")
TOKENIZER_PATH = os.path.join(ARTIFACTS_DIR, "custom_tokenizer.json")
VECTOR_STORE_NPZ = os.path.join(ARTIFACTS_DIR, "chunk_vectors.npz")
VECTOR_STORE_JSON = os.path.join(ARTIFACTS_DIR, "chunk_meta.json")

# Ollama Settings
OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:1.5b"

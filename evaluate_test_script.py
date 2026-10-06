import os
import shutil
import tempfile
import torch
import config
from loader import RepositoryLoader
from parser import CodeChunker
from dataset_builder import DatasetBuilder
from trainer import ScratchModelTrainer
from vector_store import VectorStore
from tokenizer_utils import ScratchCodeTokenizer
from embedding_model import ScratchCodeEmbeddingModel

# 30 Sample Questions with Ground Truth File Paths & Target Descriptions
SAMPLE_BENCHMARK_SUITE = [
    # Python (1-6)
    {"q": "Where is user authentication and password hashing implemented?", "expected_file": "src/auth.py"},
    {"q": "Which function computes prime numbers using sieve algorithm?", "expected_file": "src/math_ops.py"},
    {"q": "Where is the database connection pool initialized?", "expected_file": "src/db.py"},
    {"q": "Which file defines the HTTP API user login endpoint handler?", "expected_file": "src/api_routes.py"},
    {"q": "Where is payment gateway checkout logic processed?", "expected_file": "src/payment.py"},
    {"q": "Which module handles file upload and S3 storage?", "expected_file": "src/storage.py"},

    # JavaScript / TypeScript (7-12)
    {"q": "Where is the React frontend navigation bar component rendered?", "expected_file": "frontend/Navbar.jsx"},
    {"q": "Which file handles Redux user state management and actions?", "expected_file": "frontend/userStore.js"},
    {"q": "Where is JWT token decoding and storage implemented in JS?", "expected_file": "frontend/jwt_utils.js"},
    {"q": "Which function handles dark mode theme toggle switch?", "expected_file": "frontend/theme.ts"},
    {"q": "Where is WebSocket real-time chat event handler defined?", "expected_file": "frontend/chat_socket.js"},
    {"q": "Which file formats numbers into currency strings?", "expected_file": "frontend/formatters.js"},

    # Java (13-17)
    {"q": "Where is the Java Spring Security filter chain configured?", "expected_file": "java/SecurityConfig.java"},
    {"q": "Which Java class handles email notification delivery?", "expected_file": "java/EmailSenderService.java"},
    {"q": "Where is the JPA User entity model class defined?", "expected_file": "java/UserEntity.java"},
    {"q": "Which service calculates shopping cart order discounts in Java?", "expected_file": "java/DiscountCalculator.java"},
    {"q": "Where is custom exception handling for API errors defined in Java?", "expected_file": "java/GlobalExceptionHandler.java"},

    # Go (18-21)
    {"q": "Where is the Go gRPC server listener initialized?", "expected_file": "go/grpc_server.go"},
    {"q": "Which Go function implements rate limiting using token bucket algorithm?", "expected_file": "go/rate_limiter.go"},
    {"q": "Where is Redis cache connection created in Go?", "expected_file": "go/cache.go"},
    {"q": "Which Go file parses CLI command line flags?", "expected_file": "go/cli_parser.go"},

    # C++ (22-24)
    {"q": "Where is matrix multiplication optimized using multithreading in C++?", "expected_file": "cpp/matrix_math.cpp"},
    {"q": "Which C++ class manages memory allocation pool for objects?", "expected_file": "cpp/memory_pool.hpp"},
    {"q": "Where is image downsampling function implemented in C++?", "expected_file": "cpp/image_proc.cpp"},

    # PHP & Ruby (25-27)
    {"q": "Where is Laravel user session middleware defined in PHP?", "expected_file": "php/SessionMiddleware.php"},
    {"q": "Which PHP class validates email and phone form inputs?", "expected_file": "php/Validator.php"},
    {"q": "Where is Rails ActiveRecord User model defined in Ruby?", "expected_file": "ruby/user_model.rb"},

    # Markdown & Configs (28-30)
    {"q": "Where are project installation instructions and setup guides documented?", "expected_file": "README.md"},
    {"q": "Where is Docker container multi-stage build configured?", "expected_file": "Dockerfile"},
    {"q": "Which configuration file defines database environment variables?", "expected_file": ".env.example"},
]

def create_synthetic_benchmark_project(target_dir: str):
    """Creates a multi-language sample codebase matching the benchmark suite."""

    files = {
        # Python
        "src/auth.py": "def login_user(username, password):\n    \"\"\"Authenticates user with password hashing and verification.\"\"\"\n    return check_password_hash(username, password)",
        "src/math_ops.py": "def sieve_of_eratosthenes(n):\n    \"\"\"Computes prime numbers up to n using sieve algorithm.\"\"\"\n    primes = [True] * (n + 1)\n    return primes",
        "src/db.py": "class DatabasePool:\n    \"\"\"Database connection pool manager.\"\"\"\n    def connect(self):\n        print('Connecting to DB')",
        "src/api_routes.py": "def user_login_endpoint(request):\n    \"\"\"HTTP API endpoint for user login request.\"\"\"\n    return jsonify({'status': 'ok'})",
        "src/payment.py": "def process_stripe_checkout(cart_item, token):\n    \"\"\"Processes payment gateway checkout logic with Stripe API.\"\"\"\n    return stripe.Charge.create()",
        "src/storage.py": "def upload_to_s3_bucket(file_bytes, filename):\n    \"\"\"Uploads file to AWS S3 storage bucket.\"\"\"\n    return s3_client.put_object(file_bytes)",

        # JS / TS
        "frontend/Navbar.jsx": "export function Navbar() {\n    // React frontend navigation bar component\n    return <nav><h1>App Title</h1></nav>;\n}",
        "frontend/userStore.js": "export const userSlice = createSlice({\n    name: 'user',\n    initialState: { loggedIn: false },\n    reducers: { login: (state) => { state.loggedIn = true; } }\n});",
        "frontend/jwt_utils.js": "export function decodeJwtToken(token) {\n    // JWT token decoding and storage helper\n    return JSON.parse(atob(token.split('.')[1]));\n}",
        "frontend/theme.ts": "export function toggleDarkMode(isDark: boolean): void {\n    // Handles dark mode theme toggle switch\n    document.body.classList.toggle('dark', isDark);\n}",
        "frontend/chat_socket.js": "export function initWebSocketChat() {\n    // WebSocket real-time chat event handler\n    const ws = new WebSocket('ws://localhost:8080');\n}",
        "frontend/formatters.js": "export function formatCurrency(amount) {\n    // Formats numbers into localized currency strings\n    return '$' + amount.toFixed(2);\n}",

        # Java
        "java/SecurityConfig.java": "public class SecurityConfig {\n    // Java Spring Security filter chain config\n    public SecurityFilterChain configure(HttpSecurity http) {\n        return http.authorizeRequests().build();\n    }\n}",
        "java/EmailSenderService.java": "public class EmailSenderService {\n    // Handles email notification delivery\n    public void sendEmail(String to, String body) {\n        mailSender.send(message);\n    }\n}",
        "java/UserEntity.java": "public class UserEntity {\n    // JPA User entity model class\n    private Long id;\n    private String username;\n}",
        "java/DiscountCalculator.java": "public class DiscountCalculator {\n    // Calculates shopping cart order discounts\n    public double applyDiscount(double total) { return total * 0.9; }\n}",
        "java/GlobalExceptionHandler.java": "public class GlobalExceptionHandler {\n    // Custom exception handling for API errors\n    public ResponseEntity handleException(Exception ex) { return null; }\n}",

        # Go
        "go/grpc_server.go": "func InitGRPCServer(port string) {\n    // Initializes Go gRPC server listener\n    lis, _ := net.Listen('tcp', port)\n}",
        "go/rate_limiter.go": "func TokenBucketRateLimiter(clientIP string) bool {\n    // Implements rate limiting using token bucket algorithm\n    return true\n}",
        "go/cache.go": "func NewRedisConnection(url string) *redis.Client {\n    // Redis cache connection in Go\n    return redis.NewClient()\n}",
        "go/cli_parser.go": "func ParseCLIFlags() {\n    // Parses CLI command line flags\n    flag.Parse()\n}",

        # C++
        "cpp/matrix_math.cpp": "void multiply_matrices_parallel(float* A, float* B, float* C) {\n    // Matrix multiplication optimized using multithreading in C++\n}",
        "cpp/memory_pool.hpp": "class MemoryPool {\n    // Manages memory allocation pool for objects in C++\n};",
        "cpp/image_proc.cpp": "void downsample_image(unsigned char* img, int width, int height) {\n    // Image downsampling function in C++\n}",

        # PHP & Ruby
        "php/SessionMiddleware.php": "class SessionMiddleware {\n    // Laravel user session middleware in PHP\n}",
        "php/Validator.php": "class Validator {\n    // Validates email and phone form inputs in PHP\n}",
        "ruby/user_model.rb": "class User < ApplicationRecord\n    # Rails ActiveRecord User model in Ruby\nend",

        # Docs & Configs
        "README.md": "# Ask Your Codebase\nInstallation instructions and setup guides for running the local project.",
        "Dockerfile": "FROM python:3.11\n# Docker container multi-stage build configuration\nWORKDIR /app",
        ".env.example": "# Database environment variables\nDB_HOST=localhost\nDB_PORT=5432"
    }

    for rel_path, content in files.items():
        abs_path = os.path.join(target_dir, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)

def run_30_question_benchmark():
    print("=================================================================")
    print("       RUNNING 30-QUESTION BENCHMARK EVALUATION TEST             ")
    print("=================================================================\n")

    temp_dir = tempfile.mkdtemp(prefix="benchmark_codebase_")
    try:
        # Step 1: Create synthetic codebase
        create_synthetic_benchmark_project(temp_dir)

        # Step 2: Load & parse files
        loader = RepositoryLoader(temp_dir)
        files_data, root_name = loader.load()

        chunker = CodeChunker()
        all_chunks = []
        for f in files_data:
            all_chunks.extend(chunker.chunk_file(f))

        print(f"Benchmark Codebase: Ingested {len(files_data)} files into {len(all_chunks)} chunks.")

        # Step 3: Dataset pairs & split
        ds_builder = DatasetBuilder()
        pairs = ds_builder.build_repo_dataset_pairs(all_chunks)
        train_p, val_p, test_p = ds_builder.split_dataset_by_file(pairs)

        # Step 4: Train scratch model for 5 epochs
        trainer = ScratchModelTrainer()
        model, tokenizer, _ = trainer.train_and_evaluate(
            train_pairs=train_p,
            val_pairs=val_p,
            test_pairs=test_p,
            epochs=5,
            batch_size=8
        )

        # Step 5: Index vectors
        vstore = VectorStore()
        vstore.index(all_chunks, model, tokenizer)

        # Step 6: Execute 30 Benchmark Questions
        print("\n---------------- Executing 30 Sample Questions ----------------")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        top_5_hits = 0
        top_1_hits = 0

        for idx, item in enumerate(SAMPLE_BENCHMARK_SUITE, 1):
            q = item["q"]
            expected = item["expected_file"].replace("\\", "/")

            results = vstore.search(q, model, tokenizer, top_k=5, device=device)
            retrieved_files = [r["file_path"].replace("\\", "/") for r in results]

            hit_top_1 = (len(retrieved_files) > 0 and retrieved_files[0] == expected)
            hit_top_5 = (expected in retrieved_files)

            if hit_top_1: top_1_hits += 1
            if hit_top_5: top_5_hits += 1

            status_str = "[HIT Top-1]" if hit_top_1 else ("[HIT Top-5]" if hit_top_5 else "[MISS]")
            print(f"Q{idx:02d}: {q}")
            print(f"    Expected: `{expected}` | Status: {status_str}")
            if retrieved_files:
                print(f"    Top Match: `{retrieved_files[0]}` (Score: {results[0]['score']:.4f})")
            print()

        top_5_score_pct = (top_5_hits / len(SAMPLE_BENCHMARK_SUITE)) * 100.0
        top_1_score_pct = (top_1_hits / len(SAMPLE_BENCHMARK_SUITE)) * 100.0

        print("=================================================================")
        print(f"  BENCHMARK FINAL RESULTS (30 QUESTIONS):")
        print(f"  Top-1 Accuracy Score: {top_1_hits}/{len(SAMPLE_BENCHMARK_SUITE)} ({top_1_score_pct:.2f}%)")
        print(f"  Top-5 Retrieval Score: {top_5_hits}/{len(SAMPLE_BENCHMARK_SUITE)} ({top_5_score_pct:.2f}%)")
        print("=================================================================\n")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    run_30_question_benchmark()

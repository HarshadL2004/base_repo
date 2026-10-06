import os
import shutil
import tempfile
import streamlit as st
import pandas as pd
import torch

import config
from loader import RepositoryLoader
from parser import CodeChunker
from dataset_builder import DatasetBuilder
from trainer import ScratchModelTrainer
from vector_store import VectorStore
from answer_model import LocalAnswerGenerator
from tokenizer_utils import ScratchCodeTokenizer
from embedding_model import ScratchCodeEmbeddingModel

st.set_page_config(
    page_title="Ask Your Codebase (100% Local)",
    page_icon="💻",
    layout="wide"
)

st.title("💻 Ask Your Codebase")
st.markdown("### 100% Local Code Ingestion, Custom Scratch Embedding Model & Local QA")

# Sidebar Configuration
st.sidebar.header("1. Codebase Input")
source_option = st.sidebar.radio("Select Input Source:", ["ZIP File Upload", "Git Repository URL"])

target_input = None
if source_option == "ZIP File Upload":
    uploaded_zip = st.sidebar.file_uploader("Upload Project .zip", type=["zip"])
    if uploaded_zip is not None:
        temp_zip_path = os.path.join(config.TEMP_DIR, uploaded_zip.name)
        with open(temp_zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())
        target_input = temp_zip_path
else:
    git_url = st.sidebar.text_input("Git Repository URL", placeholder="https://github.com/user/repo.git")
    if git_url.strip():
        target_input = git_url.strip()

st.sidebar.header("2. Ollama Settings")
ollama_model = st.sidebar.text_input("Ollama Model Name", value=config.DEFAULT_OLLAMA_MODEL)
ollama_status_ok, status_msg = LocalAnswerGenerator.check_ollama_availability()

if ollama_status_ok:
    st.sidebar.success("✅ " + status_msg)
else:
    st.sidebar.warning("⚠️ " + status_msg + "\n\n*(Fallback Mode active: Top retrieved code chunks will be displayed directly)*")

index_button = st.sidebar.button("🚀 Index & Train Codebase", use_container_width=True)

# Initialize Session State
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "eval_metrics" not in st.session_state:
    st.session_state.eval_metrics = None

# Step 1: Ingest & Train Pipeline
if index_button:
    if not target_input:
        st.error("Please provide a valid ZIP file or Git URL before indexing.")
    else:
        with st.spinner("Step 1/5: Loading and filtering codebase files..."):
            loader = RepositoryLoader(target_input)
            files_data, root_name = loader.load()

        st.success(f"Loaded {len(files_data)} source files from `{root_name}`.")

        with st.spinner("Step 2/5: Parsing multi-language functions, classes & text blocks..."):
            chunker = CodeChunker()
            all_chunks = []
            for f in files_data:
                chunks = chunker.chunk_file(f)
                all_chunks.extend(chunks)

        st.info(f"Generated {len(all_chunks)} structural chunks across Python, JS/TS, Java, Go, C++, PHP, Ruby, etc.")

        with st.spinner("Step 3/5: Building training pairs & performing file-level 80/10/10 split..."):
            ds_builder = DatasetBuilder()
            repo_pairs = ds_builder.build_repo_dataset_pairs(all_chunks)
            train_pairs, val_pairs, test_pairs = ds_builder.split_dataset_by_file(repo_pairs)

        st.write(f"Dataset Split (by file): **Train**: {len(train_pairs)} pairs | **Val**: {len(val_pairs)} pairs | **Test**: {len(test_pairs)} pairs")

        with st.spinner("Step 4/5: Training 100% Scratch Tokenizer & PyTorch Embedding Model (InfoNCE Loss)..."):
            trainer = ScratchModelTrainer()
            model, tokenizer, eval_results = trainer.train_and_evaluate(
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                test_pairs=test_pairs,
                epochs=4
            )
            st.session_state.eval_metrics = eval_results

        with st.spinner("Step 5/5: Indexing chunk vectors into Vector Store (.npz + .json)..."):
            vstore = VectorStore()
            vstore.index(all_chunks, model, tokenizer)
            st.session_state.indexed = True

        st.success("✅ Codebase indexed and model trained successfully!")

# Display Evaluation Metrics if available
if st.session_state.eval_metrics:
    with st.expander("📊 Held-Out Test Set Evaluation Metrics (Recall & MRR)", expanded=True):
        overall = st.session_state.eval_metrics.get("overall", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("Recall@1", f"{overall.get('recall@1', 0.0)*100:.1f}%")
        col2.metric("Recall@5", f"{overall.get('recall@5', 0.0)*100:.1f}%")
        col3.metric("MRR", f"{overall.get('mrr', 0.0):.4f}")

        per_lang = st.session_state.eval_metrics.get("per_language", {})
        if per_lang:
            st.markdown("#### Per-Language Metrics Breakdown")
            df_rows = []
            for lang, m in per_lang.items():
                df_rows.append({
                    "Language": lang,
                    "Test Samples": m["sample_count"],
                    "Recall@1": f"{m['recall@1']*100:.1f}%",
                    "Recall@5": f"{m['recall@5']*100:.1f}%",
                    "MRR": f"{m['mrr']:.4f}"
                })
            st.dataframe(pd.DataFrame(df_rows), use_container_width=True)

# Step 2: Interactive Chat Interface
st.markdown("---")
st.subheader("💬 Chat With Your Codebase")

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "chunks" in msg and msg["chunks"]:
            with st.expander("🔍 View Retrieved Source Chunks"):
                for c in msg["chunks"]:
                    st.markdown(f"**File**: `{c['file_path']}` (Lines {c['start_line']}-{c['end_line']}) | **Similarity**: {c.get('score', 0.0):.4f}")
                    st.code(c["text"], language=c.get("language", "python"))

user_question = st.chat_input("Ask a question about your codebase (e.g., 'Where is authentication implemented?')...")

if user_question:
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        if not os.path.exists(config.VECTOR_STORE_NPZ) or not os.path.exists(config.MODEL_WEIGHTS_PATH):
            resp = "Please upload/specify a repository and click '🚀 Index & Train Codebase' first."
            st.markdown(resp)
            st.session_state.chat_history.append({"role": "assistant", "content": resp})
        else:
            with st.spinner("Retrieving relevant code chunks & generating response..."):
                # Load saved model & tokenizer
                tokenizer = ScratchCodeTokenizer()
                tokenizer.load(config.TOKENIZER_PATH)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = ScratchCodeEmbeddingModel().to(device)
                model.load_state_dict(torch.load(config.MODEL_WEIGHTS_PATH, map_location=device))
                model.eval()

                vstore = VectorStore()
                vstore.load()

                top_chunks = vstore.search(user_question, model, tokenizer, top_k=5, device=device)

                answer_gen = LocalAnswerGenerator(model_name=ollama_model)
                answer, used_ollama = answer_gen.generate_answer(user_question, top_chunks)

            st.markdown(answer)

            if top_chunks:
                with st.expander("🔍 View Retrieved Source Code Chunks"):
                    for c in top_chunks:
                        st.markdown(f"**File**: `{c['file_path']}` (Lines {c['start_line']}-{c['end_line']}) | **Similarity**: {c.get('score', 0.0):.4f}")
                        st.code(c["text"], language=c.get("language", "python"))

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "chunks": top_chunks
            })

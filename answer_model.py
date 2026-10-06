import json
import requests
from typing import List, Dict, Any, Tuple
import config

class LocalAnswerGenerator:
    """
    Handles local question answering using Ollama or graceful chunk fallback.
    """
    def __init__(self, model_name: str = config.DEFAULT_OLLAMA_MODEL, ollama_url: str = config.OLLAMA_URL):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")

    def check_ollama_availability() -> Tuple[bool, str]:
        """
        Checks if local Ollama service is reachable.
        """
        try:
            res = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=2)
            if res.status_code == 200:
                models = [m.get("name", "") for m in res.json().get("models", [])]
                return True, f"Ollama is running. Installed models: {', '.join(models) if models else 'None'}"
            return False, f"Ollama returned HTTP status {res.status_code}"
        except Exception as e:
            return False, f"Ollama unreachable at {config.OLLAMA_URL} ({e})"

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks into a clean prompt context block."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            fp = chunk.get("file_path", "unknown")
            start = chunk.get("start_line", 1)
            end = chunk.get("end_line", 1)
            text = chunk.get("text", "")
            lang = chunk.get("language", "")
            
            part = f"--- Context Chunk #{i} [File: {fp} | Lines: {start}-{end} | Language: {lang}] ---\n{text}\n"
            context_parts.append(part)

        return "\n".join(context_parts)

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        model_name: str = None
    ) -> Tuple[str, bool]:
        """
        Generates answer using local Ollama model if available, otherwise uses fallback.
        Returns: (answer_text, used_ollama_flag)
        """
        target_model = model_name or self.model_name
        is_available, status_msg = LocalAnswerGenerator.check_ollama_availability()

        if not retrieved_chunks:
            return "not found in the code (No matching code chunks were retrieved).", False

        if not is_available:
            return self._build_fallback_response(question, retrieved_chunks, status_msg), False

        context_str = self.format_context(retrieved_chunks)

        system_prompt = (
            "You are an expert software engineer assistant analyzing a codebase.\n"
            "INSTRUCTIONS:\n"
            "1. Answer the user's question strictly using ONLY the provided code context chunks below.\n"
            "2. Always mention the exact file path and line numbers for every code reference (e.g., `src/utils.py` lines 12-25).\n"
            "3. If the answer cannot be determined from the provided context chunks, state explicitly: 'not found in the code'.\n"
            "4. Do NOT invent functions, methods, or facts outside of the provided context."
        )

        user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}"

        payload = {
            "model": target_model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False
        }

        try:
            res = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=45)
            if res.status_code == 200:
                answer = res.json().get("response", "").strip()
                if not answer:
                    answer = "not found in the code"
                return answer, True
            else:
                fallback_msg = f"Ollama error (HTTP {res.status_code}): {res.text}"
                return self._build_fallback_response(question, retrieved_chunks, fallback_msg), False
        except Exception as e:
            fallback_msg = f"Failed to connect to Ollama: {e}"
            return self._build_fallback_response(question, retrieved_chunks, fallback_msg), False

    def _build_fallback_response(self, question: str, chunks: List[Dict[str, Any]], note: str) -> str:
        """Fallback mode when Ollama is unavailable: presents top retrieved chunks directly."""
        output = [
            f"[Fallback Mode Active] {note}\n",
            "Showing top matching code chunks retrieved from your codebase:\n"
        ]

        for i, chunk in enumerate(chunks, 1):
            fp = chunk.get("file_path", "unknown")
            start = chunk.get("start_line", 1)
            end = chunk.get("end_line", 1)
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "")
            lang = chunk.get("language", "text")

            output.append(
                f"### Match #{i} (Similarity: {score:.2f})\n"
                f"**File**: `{fp}` (Lines {start}-{end})\n"
                f"```{lang}\n{text}\n```\n"
            )

        return "\n".join(output)

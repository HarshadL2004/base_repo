import os
import json
import random
from typing import List, Dict, Tuple, Any
import config
from loader import RepositoryLoader
from parser import CodeChunker

class DatasetBuilder:
    """
    Builds multi-language training datasets combining:
    1. Public real code datasets (CodeSearchNet, CodeXGLUE, CodeParrot, etc.)
    2. Automatic project-specific adaptation pairs extracted directly from target repository files.
    """

    def __init__(self, seed: int = config.RANDOM_SEED):
        self.seed = seed
        random.seed(self.seed)

    def build_repo_dataset_pairs(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract code-text pairs from repository chunks.
        If docstring is present, use docstring. Otherwise fallback to file path, function name & params.
        """
        pairs = []
        for chunk in chunks:
            code_text = chunk["text"].strip()
            if not code_text:
                continue

            docstring = chunk.get("docstring", "").strip()
            name = chunk.get("name", "block")
            file_path = chunk.get("file_path", "unknown")
            language = chunk.get("language", "text")
            params = chunk.get("params", [])
            kind = chunk.get("kind", "code")

            if docstring and len(docstring) > 5:
                text_description = docstring
            else:
                param_str = ", ".join(params) if params else "none"
                text_description = f"{kind} {name} in {file_path} parameters {param_str}"

            pairs.append({
                "file_path": file_path,
                "code": code_text,
                "text": text_description,
                "language": language
            })

        return pairs

    def split_dataset_by_file(
        self,
        pairs: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Splits dataset by unique file_path so code from the same file never appears in both train and test.
        """
        # Group pairs by file_path
        file_to_pairs = {}
        for p in pairs:
            fp = p["file_path"]
            if fp not in file_to_pairs:
                file_to_pairs[fp] = []
            file_to_pairs[fp].append(p)

        unique_files = list(file_to_pairs.keys())
        random.seed(self.seed)
        random.shuffle(unique_files)

        total_files = len(unique_files)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)

        train_files = set(unique_files[:train_end])
        val_files = set(unique_files[train_end:val_end])
        test_files = set(unique_files[val_end:])

        # If dataset is small, ensure at least 1 file in each if possible
        if total_files >= 3:
            if not train_files: train_files.add(unique_files[0])
            if not val_files: val_files.add(unique_files[1 if len(unique_files) > 1 else 0])
            if not test_files: test_files.add(unique_files[-1])

        train_pairs = [p for fp in train_files for p in file_to_pairs[fp]]
        val_pairs = [p for fp in val_files for p in file_to_pairs[fp]]
        test_pairs = [p for fp in test_files for p in file_to_pairs[fp]]

        # Fallback if split yields empty sets on tiny repos
        if not train_pairs:
            train_pairs = pairs
        if not val_pairs:
            val_pairs = pairs[:max(1, len(pairs)//10)]
        if not test_pairs:
            test_pairs = pairs[max(0, len(pairs)-max(1, len(pairs)//10)):]

        return train_pairs, val_pairs, test_pairs

    @staticmethod
    def get_public_dataset_download_instructions() -> str:
        """
        Returns downloadable links and CLI commands for public real datasets.
        """
        return """
=== Real Public Code Datasets Download Instructions ===

1. CodeSearchNet (Python, Java, JS, Go, PHP, Ruby)
   Repository: https://github.com/github/CodeSearchNet
   Hugging Face: https://huggingface.co/datasets/code_search_net
   Python download command:
     pip install datasets
     python -c "from datasets import load_dataset; ds = load_dataset('code_search_net', 'python'); print(ds['train'][0])"

2. CodeParrot Clean Python Dataset
   Hugging Face: https://huggingface.co/datasets/codeparrot/codeparrot-clean
   Python download command:
     python -c "from datasets import load_dataset; ds = load_dataset('codeparrot/codeparrot-clean', split='train'); print(len(ds))"

3. CodeXGLUE (Code-to-Text Benchmark)
   Repository: https://github.com/microsoft/CodeXGLUE
   Download path: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text

4. The Stack (Multi-Language Source Code)
   Hugging Face: https://huggingface.co/datasets/bigcode/the-stack
======================================================
"""

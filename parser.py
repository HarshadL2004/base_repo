import ast
import re
from typing import List, Dict, Any

class CodeChunker:
    """
    Chunks code and documentation files into logical structural units (functions, classes, or paragraphs).
    Supports Python, JavaScript/TypeScript, Java, Go, C/C++, PHP, Ruby, Markdown, JSON, YAML, etc.
    """

    def __init__(self, target_line_count: int = 40):
        self.target_line_count = target_line_count

    def chunk_file(self, file_data: Dict[str, str]) -> List[Dict[str, Any]]:
        file_path = file_data["file_path"]
        content = file_data["content"]
        language = file_data["language"]

        if not content.strip():
            return []

        lines = content.splitlines()

        if language == "python":
            chunks = self._chunk_python(file_path, content, lines)
        elif language in ["javascript", "typescript", "java", "go", "cpp", "c", "php", "ruby"]:
            chunks = self._chunk_multilang_code(file_path, content, lines, language)
        else:
            chunks = self._chunk_by_paragraphs(file_path, lines, language)

        # Fallback if no structural chunks found
        if not chunks:
            chunks = self._chunk_by_paragraphs(file_path, lines, language)

        return chunks

    def _chunk_python(self, file_path: str, content: str, lines: List[str]) -> List[Dict[str, Any]]:
        chunks = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start_line = node.lineno
                    end_line = getattr(node, "end_lineno", start_line + len(ast.unparse(node).splitlines()) - 1)
                    
                    chunk_text = "\n".join(lines[start_line - 1:end_line])
                    docstring = ast.get_docstring(node) or ""
                    
                    # Extract parameter names if function
                    params = []
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        params = [arg.arg for arg in node.args.args]

                    chunks.append({
                        "file_path": file_path,
                        "start_line": start_line,
                        "end_line": end_line,
                        "language": "python",
                        "name": node.name,
                        "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                        "params": params,
                        "docstring": docstring,
                        "text": chunk_text
                    })
        except SyntaxError:
            # Fallback to multilang regex parser if syntax error occurs
            return self._chunk_multilang_code(file_path, content, lines, "python")

        return chunks

    def _chunk_multilang_code(self, file_path: str, content: str, lines: List[str], language: str) -> List[Dict[str, Any]]:
        chunks = []
        
        # Multilang regex patterns for function/class declarations
        patterns = {
            "javascript": r"(?:async\s+)?function\s+([a-zA-Z0-9_$]+)|class\s+([a-zA-Z0-9_$]+)|(?:const|let|var)\s+([a-zA-Z0-9_$]+)\s*=\s*(?:async\s*)?\(",
            "typescript": r"(?:async\s+)?function\s+([a-zA-Z0-9_$]+)|class\s+([a-zA-Z0-9_$]+)|(?:const|let|var)\s+([a-zA-Z0-9_$]+)\s*=\s*(?:async\s*)?\(",
            "java": r"(?:public|private|protected|static|\s)+[\w<>\[\]]+\s+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*\{|class\s+([a-zA-Z0-9_]+)",
            "go": r"func\s+(?:\([^)]+\)\s+)?([a-zA-Z0-9_]+)\s*\(",
            "cpp": r"(?:[\w:<>]+\s+)+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*(?:const)?\s*\{|class\s+([a-zA-Z0-9_]+)|struct\s+([a-zA-Z0-9_]+)",
            "c": r"(?:[\w]+\s+)+([a-zA-Z0-9_]+)\s*\([^)]*\)\s*\{",
            "php": r"function\s+([a-zA-Z0-9_]+)|class\s+([a-zA-Z0-9_]+)",
            "ruby": r"def\s+([a-zA-Z0-9_!?]+)|class\s+([a-zA-Z0-9_]+)",
            "python": r"def\s+([a-zA-Z0-9_]+)|class\s+([a-zA-Z0-9_]+)",
        }

        pattern = patterns.get(language, r"function\s+([a-zA-Z0-9_]+)")

        matched_indices = []
        for i, line in enumerate(lines):
            match = re.search(pattern, line)
            if match:
                # Extract matched name
                name = next((m for m in match.groups() if m is not None), "anonymous")
                matched_indices.append((i, name))

        if not matched_indices:
            return self._chunk_by_paragraphs(file_path, lines, language)

        for idx_entry, (start_idx, name) in enumerate(matched_indices):
            start_line = start_idx + 1
            if idx_entry < len(matched_indices) - 1:
                end_idx = matched_indices[idx_entry + 1][0] - 1
            else:
                end_idx = len(lines) - 1

            # Ensure minimum block size
            end_line = max(start_line, end_idx + 1)
            chunk_lines = lines[start_idx:end_line]
            chunk_text = "\n".join(chunk_lines)

            # Look for leading comments (docstring proxy) in previous 5 lines
            comment_lines = []
            for prev_i in range(max(0, start_idx - 5), start_idx):
                stripped = lines[prev_i].strip()
                if stripped.startswith("//") or stripped.startswith("#") or stripped.startswith("/*") or stripped.startswith("*"):
                    comment_lines.append(stripped.lstrip("/*# ").rstrip("*/"))
            docstring = "\n".join(comment_lines)

            chunks.append({
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "language": language,
                "name": name,
                "kind": "function",
                "params": [],
                "docstring": docstring,
                "text": chunk_text
            })

        return chunks

    def _chunk_by_paragraphs(self, file_path: str, lines: List[str], language: str) -> List[Dict[str, Any]]:
        chunks = []
        total_lines = len(lines)
        if total_lines == 0:
            return chunks

        step = self.target_line_count
        for start_idx in range(0, total_lines, step):
            end_idx = min(start_idx + step, total_lines)
            chunk_lines = lines[start_idx:end_idx]
            chunk_text = "\n".join(chunk_lines)

            if not chunk_text.strip():
                continue

            chunks.append({
                "file_path": file_path,
                "start_line": start_idx + 1,
                "end_line": end_idx,
                "language": language,
                "name": f"block_{start_idx + 1}",
                "kind": "paragraph",
                "params": [],
                "docstring": "",
                "text": chunk_text
            })

        return chunks

import os
import shutil
import zipfile
import subprocess
import tempfile
from typing import List, Dict, Tuple
import config

LANG_EXTENSION_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".php": "php",
    ".rb": "ruby",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sh": "bash",
    ".html": "html",
    ".css": "css",
}

def detect_language(file_path: str) -> str:
    """Detect language based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    return LANG_EXTENSION_MAP.get(ext, "unknown")

def is_binary_file(filepath: str) -> bool:
    """Simple check if a file is binary by attempting to read initial bytes."""
    try:
        with open(filepath, 'rb') as f:
            chunk = f.read(1024)
            if b'\0' in chunk:
                return True
            # Check ratio of printable text
            text_characters = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})
            nontext = chunk.translate(None, text_characters)
            if len(chunk) > 0 and len(nontext) / len(chunk) > 0.30:
                return True
    except Exception:
        return True
    return False

def extract_zip(zip_path: str, extract_to: str) -> str:
    """Extract a ZIP file to target directory."""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to

def clone_git_repo(git_url: str, target_dir: str) -> str:
    """Perform shallow clone of a Git repository."""
    cmd = ["git", "clone", "--depth", "1", git_url, target_dir]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return target_dir
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to clone repository: {e.stderr}")

class RepositoryLoader:
    def __init__(self, target: str):
        """
        target: Can be a path to a local zip file, local directory, or a Git URL.
        """
        self.target = target
        self.working_dir = None
        self.is_temp = False

    def load(self) -> Tuple[List[Dict[str, str]], str]:
        """
        Loads files from target repository/zip/folder.
        Returns:
            (files_data, root_dir_name)
        """
        if self.target.startswith("http://") or self.target.startswith("https://") or self.target.endswith(".git"):
            # Git URL
            repo_name = self.target.rstrip("/").split("/")[-1].replace(".git", "")
            self.working_dir = os.path.join(config.TEMP_DIR, f"git_{repo_name}")
            if os.path.exists(self.working_dir):
                shutil.rmtree(self.working_dir, ignore_errors=True)
            clone_git_repo(self.target, self.working_dir)
            self.is_temp = True
            root_name = repo_name
        elif os.path.isfile(self.target) and self.target.endswith(".zip"):
            # ZIP File
            zip_basename = os.path.splitext(os.path.basename(self.target))[0]
            self.working_dir = os.path.join(config.TEMP_DIR, f"zip_{zip_basename}")
            if os.path.exists(self.working_dir):
                shutil.rmtree(self.working_dir, ignore_errors=True)
            extract_zip(self.target, self.working_dir)
            self.is_temp = True
            root_name = zip_basename
        elif os.path.isdir(self.target):
            # Local directory
            self.working_dir = self.target
            self.is_temp = False
            root_name = os.path.basename(os.path.abspath(self.target))
        else:
            raise ValueError(f"Invalid input source: {self.target}. Must be a valid zip, directory, or git URL.")

        loaded_files = []

        for root, dirs, files in os.walk(self.working_dir):
            # Prune ignored directories in-place
            dirs[:] = [d for d in dirs if d not in config.IGNORED_DIRS and not d.startswith(".")]

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in config.IGNORED_EXTENSIONS or file.startswith("."):
                    continue

                full_path = os.path.join(root, file)

                # Skip files larger than 1 MB limit
                if os.path.getsize(full_path) > config.MAX_FILE_SIZE_BYTES:
                    continue

                if is_binary_file(full_path):
                    continue

                rel_path = os.path.relpath(full_path, self.working_dir)

                try:
                    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    
                    lang = detect_language(full_path)
                    loaded_files.append({
                        "file_path": rel_path.replace("\\", "/"),
                        "full_path": full_path,
                        "content": content,
                        "language": lang
                    })
                except Exception as e:
                    print(f"Skipping file {rel_path} due to read error: {e}")

        return loaded_files, root_name

    def cleanup(self):
        """Clean up temporary directory if created."""
        if self.is_temp and self.working_dir and os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir, ignore_errors=True)

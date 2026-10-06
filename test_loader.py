import os
import shutil
import tempfile
from loader import RepositoryLoader

def test_loader():
    print("--- Testing Phase 1: Loader ---")
    
    # Create a temporary directory structure representing a project
    temp_project = tempfile.mkdtemp(prefix="test_proj_")
    
    try:
        # Create normal python file
        py_file = os.path.join(temp_project, "main.py")
        with open(py_file, "w", encoding="utf-8") as f:
            f.write("def hello():\n    print('Hello world')\n")
            
        # Create ignored folder node_modules
        node_dir = os.path.join(temp_project, "node_modules", "package")
        os.makedirs(node_dir, exist_ok=True)
        with open(os.path.join(node_dir, "test.js"), "w", encoding="utf-8") as f:
            f.write("// Should be ignored")
            
        # Create ignored binary file
        bin_file = os.path.join(temp_project, "image.png")
        with open(bin_file, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01")
            
        # Test loading
        loader = RepositoryLoader(temp_project)
        files, root_name = loader.load()
        
        print(f"Loaded {len(files)} files from {root_name}:")
        for file in files:
            print(f" - Path: {file['file_path']}, Lang: {file['language']}, Length: {len(file['content'])} chars")
            
        assert len(files) == 1, f"Expected 1 file, got {len(files)}"
        assert files[0]['file_path'] == "main.py", f"Expected main.py, got {files[0]['file_path']}"
        print("Phase 1 Loader Test PASSED successfully!")
        
    finally:
        shutil.rmtree(temp_project, ignore_errors=True)

if __name__ == "__main__":
    test_loader()

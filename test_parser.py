from parser import CodeChunker

def test_parser():
    print("--- Testing Phase 2: Chunker & Parser ---")
    chunker = CodeChunker()

    # Test Python sample
    py_code = """
def calculate_area(width: float, height: float) -> float:
    \"\"\"Calculates the area of a rectangle.\"\"\"
    return width * height

class Circle:
    \"\"\"Represents a circle shape.\"\"\"
    def __init__(self, radius):
        self.radius = radius
"""
    py_file = {"file_path": "math_utils.py", "content": py_code, "language": "python"}
    py_chunks = chunker.chunk_file(py_file)
    print(f"Python Chunks ({len(py_chunks)}):")
    for c in py_chunks:
        print(f" - [{c['kind']}] {c['name']} (Lines {c['start_line']}-{c['end_line']}) Doc: {bool(c['docstring'])}")

    assert len(py_chunks) >= 2, f"Expected at least 2 python chunks, got {len(py_chunks)}"

    # Test JS sample
    js_code = """
// Adds two numbers together
function add(a, b) {
    return a + b;
}

class User {
    constructor(name) {
        this.name = name;
    }
}
"""
    js_file = {"file_path": "utils.js", "content": js_code, "language": "javascript"}
    js_chunks = chunker.chunk_file(js_file)
    print(f"\nJavaScript Chunks ({len(js_chunks)}):")
    for c in js_chunks:
        print(f" - {c['name']} (Lines {c['start_line']}-{c['end_line']})")

    assert len(js_chunks) >= 2, f"Expected at least 2 JS chunks, got {len(js_chunks)}"

    # Test Go sample
    go_code = """
// ComputeSum adds numbers in slice
func ComputeSum(numbers []int) int {
    sum := 0
    for _, n := range numbers {
        sum += n
    }
    return sum
}
"""
    go_file = {"file_path": "main.go", "content": go_code, "language": "go"}
    go_chunks = chunker.chunk_file(go_file)
    print(f"\nGo Chunks ({len(go_chunks)}):")
    for c in go_chunks:
        print(f" - {c['name']} (Lines {c['start_line']}-{c['end_line']})")

    assert len(go_chunks) >= 1, f"Expected at least 1 Go chunk, got {len(go_chunks)}"

    # Test Markdown paragraph sample
    md_code = """# Title

This is paragraph 1 with details.

This is paragraph 2 with instructions.
"""
    md_file = {"file_path": "README.md", "content": md_code, "language": "markdown"}
    md_chunks = chunker.chunk_file(md_file)
    print(f"\nMarkdown Chunks ({len(md_chunks)}):")
    for c in md_chunks:
        print(f" - {c['name']} (Lines {c['start_line']}-{c['end_line']})")

    assert len(md_chunks) >= 1

    print("\nPhase 2 Chunker & Parser Test PASSED successfully!")

if __name__ == "__main__":
    test_parser()

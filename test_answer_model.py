from answer_model import LocalAnswerGenerator

def test_answer_model():
    print("--- Testing Phase 6: Answer Generator & Fallback Mode ---")
    gen = LocalAnswerGenerator()

    # Test availability check
    is_avail, status = LocalAnswerGenerator.check_ollama_availability()
    print(f"Ollama status check: Available={is_avail}, Details='{status}'")

    # Sample retrieved chunks
    sample_chunks = [
        {
            "file_path": "src/auth.py",
            "start_line": 15,
            "end_line": 28,
            "language": "python",
            "name": "login_user",
            "score": 0.92,
            "text": "def login_user(username, password):\n    \"\"\"Authenticates user with password hash.\"\"\"\n    user = db.find_user(username)\n    return user.check_password(password)"
        }
    ]

    answer, used_ollama = gen.generate_answer("How does login work?", sample_chunks)
    print(f"\nGenerated Answer (Used Ollama={used_ollama}):\n{answer[:300]}...")

    assert len(answer) > 0, "Answer output must not be empty!"

    if not used_ollama:
        assert "Fallback Mode Active" in answer or "Match #1" in answer, "Fallback response should display code matches!"
        print("Fallback Mode verified successfully!")
    else:
        print("Ollama response generated successfully!")

    print("Phase 6 Answer Generator Test PASSED successfully!")

if __name__ == "__main__":
    test_answer_model()

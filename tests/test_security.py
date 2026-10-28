"""
Security and Secrets Exposure Tests.
Verifies Acceptance Gate (4): No API keys are exposed in version control or public files.
"""

from pathlib import Path


def test_gitignore_protects_secrets():
    gitignore = Path(".gitignore")
    assert gitignore.exists(), ".gitignore must exist"
    content = gitignore.read_text(encoding="utf-8")

    assert ".env" in content, ".gitignore must ignore .env files"
    assert "*.joblib" in content or "artifacts/*.joblib" in content, ".gitignore should ignore heavy binaries"


def test_no_hardcoded_keys_in_source_code():
    src_dir = Path("src")
    py_files = list(src_dir.glob("*.py"))
    assert len(py_files) > 0

    suspicious_patterns = ["AIzaSy", "sk-proj", "tvly-", "gsk_"]
    for py_file in py_files:
        text = py_file.read_text(encoding="utf-8")
        for pat in suspicious_patterns:
            assert pat not in text, f"Hardcoded API key detected in {py_file}!"

import pytest
from pathlib import Path
import hashlib
import tempfile
import os
from electrodrive.eval.governance import governance_guard, sha256_file
from electrodrive.utils.logging import JsonlLogger

@pytest.fixture
def mock_logger():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = JsonlLogger(Path(tmpdir))
        yield logger
        logger.close()

@pytest.fixture
def dummy_pdf():
    content = b"Test PDF Content for Governance P0"
    expected_hash = hashlib.sha256(content).hexdigest()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(content)
        path = Path(f.name)
    yield path, expected_hash
    if os.path.exists(path):
        os.remove(path)

def test_governance_pass(mock_logger, dummy_pdf):
    path, expected_hash = dummy_pdf
    status = governance_guard(mock_logger, path, expected_hash)
    assert status["ok"] is True
    assert status["status"] == "checked"

def test_governance_fail_mismatch(mock_logger, dummy_pdf):
    path, _ = dummy_pdf
    wrong_hash = "deadbeef" * 8
    with pytest.raises(ValueError, match="Governance SHA-256 mismatch"):
        governance_guard(mock_logger, path, wrong_hash)

def test_governance_fail_not_found(mock_logger):
    path = Path("non_existent_file_12345.pdf")
    with pytest.raises(FileNotFoundError):
        governance_guard(mock_logger, path, "a"*64)


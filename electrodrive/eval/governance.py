import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from electrodrive.utils.logging import JsonlLogger

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def governance_guard(logger: JsonlLogger, eval_pdf: Optional[Path], expected_hash: Optional[str]) -> Dict[str, Any]:
    """Fail-closed hash gate for eval-only governance."""
    rec = {
        "eval_pdf_path": str(eval_pdf) if eval_pdf else None,
        "expected_sha256": expected_hash,
        "status": "skipped",
        "actual_sha256": None,
        "ok": False,
    }
    if eval_pdf and expected_hash:
        if not Path(eval_pdf).exists():
            logger.error("Eval PDF not found.", path=str(eval_pdf))
            rec["status"] = "file_not_found"
            raise FileNotFoundError(f"Eval PDF not found at {eval_pdf}")
        actual = sha256_file(Path(eval_pdf))
        rec["actual_sha256"] = actual
        rec["status"] = "checked"
        if actual.lower() == expected_hash.lower():
            rec["ok"] = True
            logger.info("Eval-only governance PASSED.", expected=expected_hash[:16], actual=actual[:16])
        else:
            logger.error("Eval-only governance FAILED (hash mismatch).", expected=expected_hash[:16], actual=actual[:16])
            raise ValueError("Governance SHA-256 mismatch.")
    elif expected_hash or eval_pdf:
        rec["status"] = "partial"
        logger.warning("Partial governance inputs provided (need both PDF and hash).", **rec)
    else:
        logger.info("Eval-only governance skipped (no inputs).")
    return rec





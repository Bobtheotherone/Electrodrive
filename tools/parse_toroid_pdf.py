"""
Lightweight helper to extract notes from toroid.pdf using pypdf.

Usage:
    from tools.parse_toroid_pdf import load_toroid_notes
    text = load_toroid_notes()              # full text blob
    notes = load_toroid_notes(structured=True)  # dict of snippets
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

from pypdf import PdfReader


def _extract_full_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _grab_snippet(full_text: str, keywords: List[str], span: int = 400) -> str:
    text_lower = full_text.lower()
    best = None
    for kw in keywords:
        idx = text_lower.find(kw.lower())
        if idx != -1:
            best = idx if best is None else min(best, idx)
    if best is None:
        return ""
    start = max(0, best - span)
    end = min(len(full_text), best + span)
    return " ".join(full_text[start:end].split())


def load_toroid_notes(
    path: Union[str, Path] = "toroid.pdf",
    structured: bool = False,
) -> Union[str, Dict[str, str]]:
    """
    Extract notes from toroid.pdf. Returns either the full text or a dict of
    short snippets keyed by topic.
    """
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    full_text = _extract_full_text(pdf_path)
    if not structured:
        return full_text

    snippets = {
        "harmonics": _grab_snippet(
            full_text,
            ["toroidal harmonics", "associated Legendre", "m=0", "poloidal"],
        ),
        "induced_charge": _grab_snippet(
            full_text,
            ["induced charge", "charge neutrality", "total induced"],
        ),
        "asymptotics": _grab_snippet(
            full_text,
            ["asymptotic", "thin torus", "fat torus", "a/R"],
        ),
        "rational_pade": _grab_snippet(
            full_text,
            ["rational", "Pade", "model reduction", "low-rank"],
        ),
        "topology_flux": _grab_snippet(
            full_text,
            ["topology", "flux", "genus", "loop"],
        ),
    }
    snippets["full_text_len"] = str(len(full_text))
    return snippets


if __name__ == "__main__":
    notes = load_toroid_notes(structured=True)
    print("Extracted toroid.pdf notes:")
    for k, v in notes.items():
        print(f"\n[{k}]")
        print(v[:800])

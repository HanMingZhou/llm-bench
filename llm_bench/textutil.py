from __future__ import annotations


def slug(value: object) -> str:
    """Filesystem-safe slug used for run / baseline directory names."""
    text = str(value).lower()
    safe: list[str] = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        elif ch in ("/", " ", "."):
            safe.append("-")
    return "".join(safe).strip("-") or "unknown"

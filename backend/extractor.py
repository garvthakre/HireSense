"""
extractor.py
------------
Extracts raw text from PDF files using pdfplumber (primary)
with PyPDF2 as a fallback if pdfplumber fails.
"""

import pdfplumber
import PyPDF2
import io
import re
from typing import Union


def extract_text_pdfplumber(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_pypdf2(file_bytes: bytes) -> str:
    """Fallback extractor using PyPDF2."""
    text_parts = []
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_pdf(file_bytes: bytes, filename: str = "file.pdf") -> dict:
    """
    Main extraction function. Tries pdfplumber first, falls back to PyPDF2.

    Returns:
        dict with keys:
            - 'text'     : extracted raw text (str)
            - 'method'   : which library was used
            - 'filename' : original filename
            - 'success'  : bool
            - 'error'    : error message if failed
    """
    # --- Try pdfplumber first ---
    try:
        text = extract_text_pdfplumber(file_bytes)
        if text.strip():
            return {
                "text": text,
                "method": "pdfplumber",
                "filename": filename,
                "success": True,
                "error": None,
            }
    except Exception as e:
        print(f"[extractor] pdfplumber failed for '{filename}': {e}")

    # --- Fallback to PyPDF2 ---
    try:
        text = extract_text_pypdf2(file_bytes)
        if text.strip():
            return {
                "text": text,
                "method": "PyPDF2",
                "filename": filename,
                "success": True,
                "error": None,
            }
    except Exception as e:
        print(f"[extractor] PyPDF2 also failed for '{filename}': {e}")
        return {
            "text": "",
            "method": "none",
            "filename": filename,
            "success": False,
            "error": str(e),
        }

    # Both ran but returned empty text
    return {
        "text": "",
        "method": "none",
        "filename": filename,
        "success": False,
        "error": "PDF appears to be scanned/image-based with no extractable text.",
    }


def guess_candidate_name(text: str, filename: str) -> str:
    """
    Tries to guess a candidate name from the resume text or filename.
    Looks at the first 3 non-empty lines (resume usually starts with name).
    Falls back to filename stem.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()][:5]
    for line in lines:
        # A name is typically 2-4 words, no numbers, not a common header
        words = line.split()
        skip_keywords = {
            "resume", "curriculum", "vitae", "cv", "profile",
            "contact", "email", "phone", "address", "objective",
        }
        if (
            2 <= len(words) <= 4
            and not any(ch.isdigit() for ch in line)
            and not any(kw in line.lower() for kw in skip_keywords)
            and re.match(r"^[A-Za-z\s\.\-]+$", line)
        ):
            return line.title()

    # Fall back to filename without extension
    name = filename.rsplit(".", 1)[0]
    return name.replace("_", " ").replace("-", " ").title()

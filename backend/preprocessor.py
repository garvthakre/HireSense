"""
preprocessor.py
---------------
Cleans and normalises raw text extracted from PDFs.
Uses only scikit-learn's built-in stopword list so there's
no dependency on a live NLTK download.
"""

import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ── Extended stop-word list (sklearn base + common resume filler) ────────────
_EXTRA_STOP = {
    "etc", "also", "using", "used", "use", "like", "well", "able",
    "good", "strong", "excellent", "knowledge", "experience", "skill",
    "skills", "proficiency", "proficient", "familiarity", "familiar",
    "understanding", "understanding", "bachelor", "master", "degree",
    "university", "college", "institute", "cgpa", "gpa", "grade",
    "currently", "previous", "present", "responsibilities", "responsible",
    "worked", "working", "work", "year", "years", "month", "months",
}

STOP_WORDS = ENGLISH_STOP_WORDS.union(_EXTRA_STOP)


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline:
      1. Lowercase
      2. Remove URLs, emails, phone numbers
      3. Remove punctuation / special chars (keep alphanumeric + spaces)
      4. Collapse whitespace
      5. Remove stop words
      6. Remove single-character tokens
    Returns a clean, single-line string.
    """
    if not text:
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove emails
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove phone numbers (various formats)
    text = re.sub(r"[\+\(]?[1-9][0-9\s\.\-\(\)]{6,}[0-9]", " ", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Tokenise
    tokens = text.split()

    # Filter stop words and short tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    return " ".join(tokens)


def extract_sections(raw_text: str) -> dict:
    """
    Attempts to split resume/JD text into logical sections.
    Returns a dict of section_name → content.
    Useful for targeted skill extraction later.
    """
    section_headers = [
        "education", "experience", "skills", "projects",
        "certifications", "achievements", "summary", "objective",
        "internship", "internships", "technologies", "tools",
        "requirements", "responsibilities", "qualifications",
    ]

    # Build pattern that matches any header at the start of a line
    pattern = r"(?im)^(" + "|".join(section_headers) + r")[:\s]*$"
    parts = re.split(pattern, raw_text, flags=re.IGNORECASE | re.MULTILINE)

    sections = {"full": raw_text}
    current_section = "intro"
    buffer = []

    for part in parts:
        part_stripped = part.strip().lower()
        if part_stripped in section_headers:
            if buffer:
                sections[current_section] = "\n".join(buffer).strip()
            current_section = part_stripped
            buffer = []
        else:
            buffer.append(part)

    if buffer:
        sections[current_section] = "\n".join(buffer).strip()

    return sections


def preprocess_for_tfidf(text: str) -> str:
    """
    Light preprocessing specifically for TF-IDF vectorisation.
    Keeps more content than full clean (e.g. numbers) so that
    version numbers like 'python3' or 'node16' are preserved.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    # Keep alphanumeric (including version numbers like python3, node16)
    text = re.sub(r"[^a-z0-9\s\+\#]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

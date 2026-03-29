"""
skill_matcher.py
----------------
Extracts skills from text by matching against a curated
keyword list grouped into categories.
Compares JD skills vs resume skills to produce match/gap reports.
"""

from __future__ import annotations
import re
from typing import NamedTuple


# ── Master skill taxonomy ────────────────────────────────────────────────────

SKILL_TAXONOMY: dict[str, list[str]] = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c", "c++", "c#",
        "go", "golang", "rust", "kotlin", "swift", "php", "ruby", "scala",
        "r", "matlab", "perl", "bash", "shell", "powershell", "dart",
    ],
    "Web Frontend": [
        "react", "nextjs", "next.js", "vuejs", "vue", "angular", "html",
        "css", "tailwind", "bootstrap", "sass", "scss", "webpack", "vite",
        "jquery", "redux", "zustand", "svelte",
    ],
    "Web Backend": [
        "nodejs", "node.js", "express", "fastapi", "flask", "django",
        "spring", "springboot", "laravel", "rails", "asp.net", "graphql",
        "rest", "restful", "api", "microservices",
    ],
    "Databases": [
        "mysql", "postgresql", "postgres", "sqlite", "mongodb", "redis",
        "cassandra", "oracle", "mssql", "dynamodb", "firebase", "supabase",
        "neo4j", "elasticsearch",
    ],
    "Machine Learning / AI": [
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
        "sklearn", "xgboost", "lightgbm", "huggingface", "transformers",
        "bert", "gpt", "llm", "opencv", "pandas", "numpy", "matplotlib",
        "seaborn", "plotly",
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
        "jenkins", "ci/cd", "github actions", "terraform", "ansible",
        "linux", "nginx", "apache", "heroku", "vercel", "netlify",
    ],
    "Data Engineering": [
        "spark", "hadoop", "kafka", "airflow", "dbt", "snowflake",
        "bigquery", "data pipeline", "etl", "data warehouse",
    ],
    "Version Control & Tools": [
        "git", "github", "gitlab", "bitbucket", "jira", "confluence",
        "postman", "swagger", "vs code", "intellij",
    ],
    "Mobile": [
        "android", "ios", "react native", "flutter", "xamarin",
    ],
    "Soft Skills": [
        "communication", "teamwork", "leadership", "problem solving",
        "analytical", "critical thinking", "time management",
        "collaboration", "adaptability",
    ],
}

# Flatten to a lookup: skill_text → category
_SKILL_TO_CATEGORY: dict[str, str] = {}
for _cat, _skills in SKILL_TAXONOMY.items():
    for _skill in _skills:
        _SKILL_TO_CATEGORY[_skill.lower()] = _cat

# All skill keywords sorted longest-first to avoid partial matches
_ALL_SKILLS_SORTED: list[str] = sorted(
    _SKILL_TO_CATEGORY.keys(), key=len, reverse=True
)


# ── Core extraction ──────────────────────────────────────────────────────────

def extract_skills(text: str) -> dict[str, list[str]]:
    """
    Scan `text` for known skill keywords.

    Returns:
        dict of category → [matched skill strings]
    """
    text_lower = text.lower()
    found: dict[str, list[str]] = {}
    matched_spans: list[tuple[int, int]] = []

    for skill in _ALL_SKILLS_SORTED:
        # Word-boundary match to avoid partial hits (e.g. "r" inside "array")
        pattern = r"(?<![a-z0-9\+\#])" + re.escape(skill) + r"(?![a-z0-9\+\#])"
        for m in re.finditer(pattern, text_lower):
            start, end = m.start(), m.end()
            # Skip if already covered by a longer match
            if any(s <= start and end <= e for s, e in matched_spans):
                continue
            matched_spans.append((start, end))
            cat = _SKILL_TO_CATEGORY[skill]
            found.setdefault(cat, [])
            if skill not in found[cat]:
                found[cat].append(skill)

    return found


def flatten_skills(categorised: dict[str, list[str]]) -> set[str]:
    """Flatten category→skills dict into a flat set of skill strings."""
    result: set[str] = set()
    for skills in categorised.values():
        result.update(skills)
    return result


# ── Comparison ───────────────────────────────────────────────────────────────

class SkillMatchResult(NamedTuple):
    matched_skills: list[str]           # skills in both JD and resume
    missing_skills: list[str]           # skills in JD but NOT in resume
    extra_skills: list[str]             # skills in resume but NOT required by JD
    matched_by_category: dict[str, list[str]]
    missing_by_category: dict[str, list[str]]
    skill_match_score: float            # 0.0 – 1.0


def compare_skills(jd_text: str, resume_text: str) -> SkillMatchResult:
    """
    Compare skills extracted from a job description against a resume.

    Returns a SkillMatchResult with match/gap detail.
    """
    jd_skills_cat    = extract_skills(jd_text)
    resume_skills_cat = extract_skills(resume_text)

    jd_flat     = flatten_skills(jd_skills_cat)
    resume_flat = flatten_skills(resume_skills_cat)

    matched = sorted(jd_flat & resume_flat)
    missing = sorted(jd_flat - resume_flat)
    extra   = sorted(resume_flat - jd_flat)

    # Category-wise breakdown
    matched_by_cat: dict[str, list[str]] = {}
    missing_by_cat: dict[str, list[str]] = {}

    for cat, skills in jd_skills_cat.items():
        res_cat_skills = set(resume_skills_cat.get(cat, []))
        jd_cat_skills  = set(skills)
        m = sorted(jd_cat_skills & res_cat_skills)
        miss = sorted(jd_cat_skills - res_cat_skills)
        if m:
            matched_by_cat[cat] = m
        if miss:
            missing_by_cat[cat] = miss

    # Skill match score: Jaccard on JD skills
    skill_score = len(matched) / len(jd_flat) if jd_flat else 0.0

    return SkillMatchResult(
        matched_skills=matched,
        missing_skills=missing,
        extra_skills=extra,
        matched_by_category=matched_by_cat,
        missing_by_category=missing_by_cat,
        skill_match_score=round(skill_score, 4),
    )

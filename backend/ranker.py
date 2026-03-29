"""
ranker.py
---------
Sorts ResumeAnalysis results and assigns ranks with tie-handling,
score bands, and recommendation labels.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from nlp_engine import ResumeAnalysis


# ── Score band thresholds ────────────────────────────────────────────────────
BAND_THRESHOLDS = [
    (80, "Excellent Match",   "green"),
    (65, "Strong Match",      "blue"),
    (50, "Moderate Match",    "yellow"),
    (35, "Partial Match",     "orange"),
    ( 0, "Low Match",         "red"),
]


@dataclass
class RankedCandidate:
    rank: int
    candidate_name: str
    filename: str
    combined_score: float
    tfidf_score: float
    skill_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    extra_skills: list[str]
    matched_by_category: dict[str, list[str]]
    missing_by_category: dict[str, list[str]]
    score_band: str
    band_color: str
    recommendation: str


def _get_band(score: float) -> tuple[str, str]:
    for threshold, label, color in BAND_THRESHOLDS:
        if score >= threshold:
            return label, color
    return "Low Match", "red"


def _build_recommendation(score: float, missing: list[str]) -> str:
    band_label, _ = _get_band(score)
    if score >= 80:
        return "Highly recommended for interview."
    elif score >= 65:
        top_missing = ", ".join(missing[:3]) if missing else "none"
        return f"Good candidate. Consider upskilling in: {top_missing}."
    elif score >= 50:
        top_missing = ", ".join(missing[:4]) if missing else "none"
        return f"Moderate fit. Key gaps: {top_missing}."
    elif score >= 35:
        top_missing = ", ".join(missing[:5]) if missing else "none"
        return f"Partial match. Significant gaps: {top_missing}."
    else:
        return "Does not sufficiently match the job requirements."


def rank_candidates(
    analyses: list[ResumeAnalysis],
    sort_by: str = "combined_score",   # "combined_score" | "tfidf_score" | "skill_score"
) -> list[RankedCandidate]:
    """
    Takes raw NLP analysis results and returns a sorted, ranked list.

    Tie-breaking order: combined_score → skill_score → tfidf_score → name (alpha)
    """
    valid_sort_keys = {"combined_score", "tfidf_score", "skill_score"}
    if sort_by not in valid_sort_keys:
        sort_by = "combined_score"

    # Sort descending with tie-breakers
    sorted_analyses = sorted(
        analyses,
        key=lambda a: (
            getattr(a, sort_by),
            a.skill_score,
            a.tfidf_score,
            a.candidate_name.lower(),
        ),
        reverse=True,
    )

    ranked: list[RankedCandidate] = []
    prev_score: Optional[float] = None
    prev_rank: int = 0

    for i, analysis in enumerate(sorted_analyses, start=1):
        current_score = round(getattr(analysis, sort_by), 2)

        # Assign same rank for tied scores
        if current_score == prev_score:
            rank = prev_rank
        else:
            rank = i
            prev_rank = i

        prev_score = current_score

        band_label, band_color = _get_band(analysis.combined_score)
        recommendation = _build_recommendation(
            analysis.combined_score,
            analysis.skill_match.missing_skills,
        )

        ranked.append(
            RankedCandidate(
                rank               = rank,
                candidate_name     = analysis.candidate_name,
                filename           = analysis.filename,
                combined_score     = analysis.combined_score,
                tfidf_score        = analysis.tfidf_score,
                skill_score        = analysis.skill_score,
                matched_skills     = analysis.skill_match.matched_skills,
                missing_skills     = analysis.skill_match.missing_skills,
                extra_skills       = analysis.skill_match.extra_skills,
                matched_by_category= analysis.skill_match.matched_by_category,
                missing_by_category= analysis.skill_match.missing_by_category,
                score_band         = band_label,
                band_color         = band_color,
                recommendation     = recommendation,
            )
        )

    return ranked


def get_session_summary(ranked: list[RankedCandidate]) -> dict:
    """Returns aggregate stats for the session — useful for dashboard headers."""
    if not ranked:
        return {}

    scores = [c.combined_score for c in ranked]
    band_counts: dict[str, int] = {}
    for c in ranked:
        band_counts[c.score_band] = band_counts.get(c.score_band, 0) + 1

    return {
        "total_candidates": len(ranked),
        "highest_score":    round(max(scores), 2),
        "lowest_score":     round(min(scores), 2),
        "average_score":    round(sum(scores) / len(scores), 2),
        "top_candidate":    ranked[0].candidate_name if ranked else None,
        "band_distribution": band_counts,
    }

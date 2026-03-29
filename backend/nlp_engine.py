"""
nlp_engine.py
-------------
Core NLP analysis module.

Responsibilities:
  - TF-IDF vectorisation of job description + all resumes
  - Cosine similarity computation (resume vs JD)
  - Combined scoring: blends TF-IDF similarity + skill match score
  - Returns per-resume NLP analysis results
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessor import preprocess_for_tfidf, clean_text
from skill_matcher import compare_skills, SkillMatchResult


# ── Result containers ────────────────────────────────────────────────────────

@dataclass
class ResumeAnalysis:
    """Analysis result for a single resume against one job description."""
    filename: str
    candidate_name: str

    # Similarity scores (0–100)
    tfidf_score: float          # Raw TF-IDF cosine similarity %
    skill_score: float          # Keyword skill match %
    combined_score: float       # Weighted blend (primary ranking key)

    # Skill details
    skill_match: SkillMatchResult

    # Raw text stored for debugging / re-analysis
    raw_text: str = field(repr=False, default="")

    # Optional: per-section scores (future use)
    section_scores: dict = field(default_factory=dict)


@dataclass
class SessionAnalysis:
    """Full analysis result for one upload session."""
    session_id: str
    jd_filename: str
    total_resumes: int
    results: list[ResumeAnalysis] = field(default_factory=list)


# ── Weights for combined score ───────────────────────────────────────────────
#   Tune these to shift emphasis between semantic similarity and keyword match.
TFIDF_WEIGHT  = 0.55   # 55 % weight → semantic / contextual overlap
SKILL_WEIGHT  = 0.45   # 45 % weight → hard skill keyword overlap


# ── Main engine ─────────────────────────────────────────────────────────────

class NLPEngine:
    """
    Orchestrates TF-IDF vectorisation and scoring for one analysis session.

    Usage:
        engine = NLPEngine()
        results = engine.analyse(jd_text, resumes)
    """

    def __init__(
        self,
        tfidf_weight: float = TFIDF_WEIGHT,
        skill_weight: float = SKILL_WEIGHT,
        ngram_range: tuple = (1, 2),
        max_features: int = 8000,
    ):
        self.tfidf_weight = tfidf_weight
        self.skill_weight = skill_weight
        self.ngram_range  = ngram_range
        self.max_features = max_features

        # Vectoriser is re-created per session so it fits the current corpus
        self._vectorizer: Optional[TfidfVectorizer] = None

    # ── Public API ───────────────────────────────────────────────────────────

    def analyse(
        self,
        jd_text: str,
        resumes: list[dict],   # [{"filename": str, "candidate_name": str, "text": str}]
    ) -> list[ResumeAnalysis]:
        """
        Analyse all resumes against the job description.

        Args:
            jd_text  : Raw extracted text of the job description.
            resumes  : List of dicts, each with 'filename', 'candidate_name', 'text'.

        Returns:
            List of ResumeAnalysis objects (unsorted).
        """
        if not resumes:
            return []

        # 1. Preprocess all texts for TF-IDF
        jd_clean = preprocess_for_tfidf(jd_text)
        resume_cleans = [preprocess_for_tfidf(r["text"]) for r in resumes]

        # 2. Fit TF-IDF on the full corpus (JD + all resumes)
        corpus = [jd_clean] + resume_cleans
        self._vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,          # log(1+tf) to dampen high-freq terms
            min_df=1,
        )
        tfidf_matrix = self._vectorizer.fit_transform(corpus)

        jd_vector       = tfidf_matrix[0]          # first row = JD
        resume_vectors  = tfidf_matrix[1:]          # remaining rows = resumes

        # 3. Cosine similarities
        cos_scores = cosine_similarity(jd_vector, resume_vectors).flatten()

        # 4. Build individual results
        results: list[ResumeAnalysis] = []
        for idx, resume in enumerate(resumes):
            tfidf_pct  = round(float(cos_scores[idx]) * 100, 2)

            # Skill matching (uses raw text for better keyword detection)
            skill_match = compare_skills(jd_text, resume["text"])
            skill_pct   = round(skill_match.skill_match_score * 100, 2)

            # Weighted combined score
            combined = round(
                self.tfidf_weight * tfidf_pct + self.skill_weight * skill_pct, 2
            )

            results.append(
                ResumeAnalysis(
                    filename        = resume["filename"],
                    candidate_name  = resume["candidate_name"],
                    tfidf_score     = tfidf_pct,
                    skill_score     = skill_pct,
                    combined_score  = combined,
                    skill_match     = skill_match,
                    raw_text        = resume["text"],
                )
            )

        return results

    def get_top_tfidf_terms(self, top_n: int = 20) -> list[str]:
        """
        Returns the most important TF-IDF feature terms from the last fit.
        Useful for debugging / explaining what keywords drove the scores.
        """
        if self._vectorizer is None:
            return []
        feature_names = self._vectorizer.get_feature_names_out()
        return list(feature_names[:top_n])


# ── Convenience helper ───────────────────────────────────────────────────────

def run_analysis(jd_text: str, resumes: list[dict]) -> list[ResumeAnalysis]:
    """
    One-shot helper. Creates an NLPEngine, runs analysis, returns results.
    Equivalent to:  NLPEngine().analyse(jd_text, resumes)
    """
    engine = NLPEngine()
    return engine.analyse(jd_text, resumes)

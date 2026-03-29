"""
database.py
-----------
SQLAlchemy ORM models and session management using SQLite.
Stores analysis sessions and per-candidate results for history / re-viewing.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    DateTime, Text, ForeignKey, Boolean,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

# ── Engine ───────────────────────────────────────────────────────────────────
DATABASE_URL = "sqlite:///./resume_analysis.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite + FastAPI
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── ORM Models ───────────────────────────────────────────────────────────────

class AnalysisSession(Base):
    """
    One analysis session = one job description + N resumes uploaded together.
    """
    __tablename__ = "analysis_sessions"

    id              = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at      = Column(DateTime, default=datetime.utcnow)
    jd_filename     = Column(String, nullable=False)
    jd_text_snippet = Column(Text, nullable=True)   # First 500 chars of JD for preview
    total_resumes   = Column(Integer, default=0)
    top_candidate   = Column(String, nullable=True)
    average_score   = Column(Float, nullable=True)
    is_complete     = Column(Boolean, default=False)

    # Relationship to results
    results = relationship("CandidateResult", back_populates="session",
                           cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Session id={self.id[:8]} jd={self.jd_filename}>"


class CandidateResult(Base):
    """
    Per-candidate analysis result linked to a session.
    Skill lists are stored as JSON strings.
    """
    __tablename__ = "candidate_results"

    id             = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id     = Column(String, ForeignKey("analysis_sessions.id"), nullable=False)
    rank           = Column(Integer, nullable=False)
    candidate_name = Column(String, nullable=False)
    filename       = Column(String, nullable=False)

    combined_score = Column(Float, nullable=False)
    tfidf_score    = Column(Float, nullable=False)
    skill_score    = Column(Float, nullable=False)
    score_band     = Column(String, nullable=False)
    band_color     = Column(String, nullable=False)
    recommendation = Column(Text, nullable=True)

    # JSON-serialised skill lists
    matched_skills          = Column(Text, default="[]")
    missing_skills          = Column(Text, default="[]")
    extra_skills            = Column(Text, default="[]")
    matched_by_category     = Column(Text, default="{}")
    missing_by_category     = Column(Text, default="{}")

    session = relationship("AnalysisSession", back_populates="results")

    # ── JSON helpers ─────────────────────────────────────────────────────────
    def set_skills(
        self,
        matched: list, missing: list, extra: list,
        matched_cat: dict, missing_cat: dict,
    ):
        self.matched_skills      = json.dumps(matched)
        self.missing_skills      = json.dumps(missing)
        self.extra_skills        = json.dumps(extra)
        self.matched_by_category = json.dumps(matched_cat)
        self.missing_by_category = json.dumps(missing_cat)

    def get_matched_skills(self)  -> list:  return json.loads(self.matched_skills or "[]")
    def get_missing_skills(self)  -> list:  return json.loads(self.missing_skills or "[]")
    def get_extra_skills(self)    -> list:  return json.loads(self.extra_skills or "[]")
    def get_matched_by_cat(self)  -> dict:  return json.loads(self.matched_by_category or "{}")
    def get_missing_by_cat(self)  -> dict:  return json.loads(self.missing_by_category or "{}")

    def __repr__(self):
        return f"<Candidate {self.candidate_name} rank={self.rank} score={self.combined_score}>"


# ── DB setup ─────────────────────────────────────────────────────────────────

def create_tables():
    """Create all tables if they don't exist. Call once at startup."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that yields a DB session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── CRUD helpers ─────────────────────────────────────────────────────────────

def save_session(
    db: Session,
    session_id: str,
    jd_filename: str,
    jd_text: str,
    ranked_candidates: list,   # list[RankedCandidate]
    summary: dict,
) -> AnalysisSession:
    """Persist a completed analysis session and all candidate results."""

    session_row = AnalysisSession(
        id              = session_id,
        jd_filename     = jd_filename,
        jd_text_snippet = jd_text[:500],
        total_resumes   = summary.get("total_candidates", 0),
        top_candidate   = summary.get("top_candidate"),
        average_score   = summary.get("average_score"),
        is_complete     = True,
    )
    db.add(session_row)

    for candidate in ranked_candidates:
        result_row = CandidateResult(
            session_id     = session_id,
            rank           = candidate.rank,
            candidate_name = candidate.candidate_name,
            filename       = candidate.filename,
            combined_score = candidate.combined_score,
            tfidf_score    = candidate.tfidf_score,
            skill_score    = candidate.skill_score,
            score_band     = candidate.score_band,
            band_color     = candidate.band_color,
            recommendation = candidate.recommendation,
        )
        result_row.set_skills(
            matched     = candidate.matched_skills,
            missing     = candidate.missing_skills,
            extra       = candidate.extra_skills,
            matched_cat = candidate.matched_by_category,
            missing_cat = candidate.missing_by_category,
        )
        db.add(result_row)

    db.commit()
    db.refresh(session_row)
    return session_row


def get_all_sessions(db: Session) -> list[AnalysisSession]:
    return db.query(AnalysisSession).order_by(AnalysisSession.created_at.desc()).all()


def get_session_by_id(db: Session, session_id: str) -> Optional[AnalysisSession]:
    return db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()


def get_candidate_by_id(db: Session, candidate_id: str) -> Optional[CandidateResult]:
    return db.query(CandidateResult).filter(CandidateResult.id == candidate_id).first()


def get_results_for_session(db: Session, session_id: str) -> list[CandidateResult]:
    return (
        db.query(CandidateResult)
        .filter(CandidateResult.session_id == session_id)
        .order_by(CandidateResult.rank)
        .all()
    )


def delete_session(db: Session, session_id: str) -> bool:
    session_row = get_session_by_id(db, session_id)
    if not session_row:
        return False
    db.delete(session_row)
    db.commit()
    return True

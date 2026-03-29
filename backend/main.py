"""
main.py
-------
FastAPI application — entry point.

Endpoints:
  POST /api/analyse          Upload JD + resumes, run full NLP analysis
  GET  /api/results/{id}     Get ranked results for a session
  GET  /api/student/{id}     Get detailed report for one candidate
  GET  /api/history          List all past sessions
  DELETE /api/session/{id}   Delete a session

Run with:
  uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import (
    create_tables, get_db,
    save_session, get_all_sessions, get_session_by_id,
    get_results_for_session, get_candidate_by_id, delete_session,
)
from extractor import extract_text_from_pdf, guess_candidate_name
from nlp_engine import run_analysis
from ranker import rank_candidates, get_session_summary

from sqlalchemy.orm import Session

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Resume Analysis API",
    description="NLP-based multi-student resume vs job description analysis with ranking",
    version="1.0.0",
)

# Allow Next.js frontend on port 3000 during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create DB tables on startup
@app.on_event("startup")
def startup():
    create_tables()
    print("✅  Database tables ready.")


# ── Pydantic response schemas ────────────────────────────────────────────────

class CandidateShort(BaseModel):
    id: str
    rank: int
    candidate_name: str
    filename: str
    combined_score: float
    tfidf_score: float
    skill_score: float
    score_band: str
    band_color: str
    recommendation: str

class CandidateDetailed(CandidateShort):
    matched_skills: list[str]
    missing_skills: list[str]
    extra_skills: list[str]
    matched_by_category: dict[str, list[str]]
    missing_by_category: dict[str, list[str]]

class SessionShort(BaseModel):
    id: str
    created_at: str
    jd_filename: str
    total_resumes: int
    top_candidate: Optional[str]
    average_score: Optional[float]

class AnalyseResponse(BaseModel):
    session_id: str
    jd_filename: str
    total_resumes: int
    summary: dict
    ranked_candidates: list[CandidateDetailed]

class HistoryResponse(BaseModel):
    sessions: list[SessionShort]

class ResultsResponse(BaseModel):
    session_id: str
    jd_filename: str
    summary: dict
    candidates: list[CandidateDetailed]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Resume Analysis API is running 🚀", "docs": "/docs"}


@app.post("/api/analyse", response_model=AnalyseResponse)
async def analyse(
    jd_file: UploadFile = File(..., description="Job Description PDF"),
    resume_files: list[UploadFile] = File(..., description="Resume PDFs (multiple)"),
    sort_by: str = Form(default="combined_score"),
    db: Session = Depends(get_db),
):
    """
    Main analysis endpoint.
    Accepts one JD PDF and multiple resume PDFs.
    Returns a ranked list of candidates with skill match details.
    """
    # ── 1. Extract JD text ──────────────────────────────────────────────────
    jd_bytes = await jd_file.read()
    jd_result = extract_text_from_pdf(jd_bytes, jd_file.filename)
    if not jd_result["success"]:
        raise HTTPException(
            status_code=422,
            detail=f"Could not extract text from JD '{jd_file.filename}': {jd_result['error']}"
        )
    jd_text = jd_result["text"]

    # ── 2. Extract resume texts ─────────────────────────────────────────────
    if not resume_files:
        raise HTTPException(status_code=400, detail="No resume files uploaded.")

    resumes = []
    failed_files = []
    for rf in resume_files:
        rb = await rf.read()
        res = extract_text_from_pdf(rb, rf.filename)
        if res["success"]:
            name = guess_candidate_name(res["text"], rf.filename)
            resumes.append({
                "filename": rf.filename,
                "candidate_name": name,
                "text": res["text"],
            })
        else:
            failed_files.append(rf.filename)

    if not resumes:
        raise HTTPException(
            status_code=422,
            detail=f"All uploaded resume files failed text extraction: {failed_files}"
        )

    # ── 3. Run NLP analysis ─────────────────────────────────────────────────
    analyses = run_analysis(jd_text, resumes)

    # ── 4. Rank candidates ──────────────────────────────────────────────────
    ranked = rank_candidates(analyses, sort_by=sort_by)
    summary = get_session_summary(ranked)

    # ── 5. Persist to DB ────────────────────────────────────────────────────
    session_id = str(uuid.uuid4())
    save_session(db, session_id, jd_file.filename, jd_text, ranked, summary)

    # ── 6. Build response ───────────────────────────────────────────────────
    # Fetch saved rows so we have the DB-generated IDs
    saved_results = get_results_for_session(db, session_id)
    id_map = {r.filename: r.id for r in saved_results}

    candidates_out = [
        CandidateDetailed(
            id             = id_map.get(c.filename, ""),
            rank           = c.rank,
            candidate_name = c.candidate_name,
            filename       = c.filename,
            combined_score = c.combined_score,
            tfidf_score    = c.tfidf_score,
            skill_score    = c.skill_score,
            score_band     = c.score_band,
            band_color     = c.band_color,
            recommendation = c.recommendation,
            matched_skills = c.matched_skills,
            missing_skills = c.missing_skills,
            extra_skills   = c.extra_skills,
            matched_by_category = c.matched_by_category,
            missing_by_category = c.missing_by_category,
        )
        for c in ranked
    ]

    return AnalyseResponse(
        session_id      = session_id,
        jd_filename     = jd_file.filename,
        total_resumes   = len(resumes),
        summary         = summary,
        ranked_candidates = candidates_out,
    )


@app.get("/api/results/{session_id}", response_model=ResultsResponse)
def get_results(session_id: str, db: Session = Depends(get_db)):
    """Retrieve full ranked results for an existing session."""
    session = get_session_by_id(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    db_results = get_results_for_session(db, session_id)

    candidates = [
        CandidateDetailed(
            id             = r.id,
            rank           = r.rank,
            candidate_name = r.candidate_name,
            filename       = r.filename,
            combined_score = r.combined_score,
            tfidf_score    = r.tfidf_score,
            skill_score    = r.skill_score,
            score_band     = r.score_band,
            band_color     = r.band_color,
            recommendation = r.recommendation,
            matched_skills = r.get_matched_skills(),
            missing_skills = r.get_missing_skills(),
            extra_skills   = r.get_extra_skills(),
            matched_by_category = r.get_matched_by_cat(),
            missing_by_category = r.get_missing_by_cat(),
        )
        for r in db_results
    ]

    summary = {
        "total_candidates": session.total_resumes,
        "top_candidate":    session.top_candidate,
        "average_score":    session.average_score,
    }

    return ResultsResponse(
        session_id  = session_id,
        jd_filename = session.jd_filename,
        summary     = summary,
        candidates  = candidates,
    )


@app.get("/api/student/{candidate_id}", response_model=CandidateDetailed)
def get_student(candidate_id: str, db: Session = Depends(get_db)):
    """Retrieve detailed report for one specific candidate."""
    row = get_candidate_by_id(db, candidate_id)
    if not row:
        raise HTTPException(status_code=404, detail="Candidate not found.")

    return CandidateDetailed(
        id             = row.id,
        rank           = row.rank,
        candidate_name = row.candidate_name,
        filename       = row.filename,
        combined_score = row.combined_score,
        tfidf_score    = row.tfidf_score,
        skill_score    = row.skill_score,
        score_band     = row.score_band,
        band_color     = row.band_color,
        recommendation = row.recommendation,
        matched_skills = row.get_matched_skills(),
        missing_skills = row.get_missing_skills(),
        extra_skills   = row.get_extra_skills(),
        matched_by_category = row.get_matched_by_cat(),
        missing_by_category = row.get_missing_by_cat(),
    )


@app.get("/api/history", response_model=HistoryResponse)
def get_history(db: Session = Depends(get_db)):
    """List all past analysis sessions."""
    sessions = get_all_sessions(db)
    return HistoryResponse(
        sessions=[
            SessionShort(
                id            = s.id,
                created_at    = s.created_at.isoformat(),
                jd_filename   = s.jd_filename,
                total_resumes = s.total_resumes,
                top_candidate = s.top_candidate,
                average_score = s.average_score,
            )
            for s in sessions
        ]
    )


@app.delete("/api/session/{session_id}")
def remove_session(session_id: str, db: Session = Depends(get_db)):
    """Delete a session and all its candidate results."""
    success = delete_session(db, session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": f"Session {session_id} deleted successfully."}

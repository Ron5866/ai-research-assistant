from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


# ── Auth ──
class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserOut(BaseModel):
    id: int
    name: str
    email: str
    created_at: datetime
    class Config:
        from_attributes = True


# ── Papers ──
class PaperOut(BaseModel):
    id: int
    original_name: str
    chunk_count: int
    uploaded_at: datetime
    title: Optional[str]
    authors: Optional[str]
    year: Optional[str]
    class Config:
        from_attributes = True


# ── Insights ──
class InsightsOut(BaseModel):
    title: Optional[str]
    authors: Optional[str]
    year: Optional[str]
    problem: Optional[str]
    methodology: Optional[str]
    dataset: Optional[str]
    contributions: Optional[str]
    results: Optional[str]
    limitations: Optional[str]


# ── RAG ──
class AskRequest(BaseModel):
    question: str
    chat_history: list[dict] = []  # [{"role": "user/assistant", "content": "..."}]

class AskResponse(BaseModel):
    answer: str
    sources: list[str] = []  # page snippets used

class SummaryResponse(BaseModel):
    summary: str


# ── Compare ──
class CompareRequest(BaseModel):
    paper_id_1: int
    paper_id_2: int
    aspect: str = "methodology"  # what to compare
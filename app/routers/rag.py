from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_user
from app import models, schemas
from app.services import rag_service, insights_service

router = APIRouter(prefix="/papers", tags=["RAG"])


def get_paper_or_404(paper_id, user_id, db):
    paper = db.query(models.Paper)\
               .filter(models.Paper.id == paper_id,
                       models.Paper.user_id == user_id)\
               .first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper


@router.post("/{paper_id}/ask", response_model=schemas.AskResponse)
def ask_question(
    paper_id: int,
    body: schemas.AskRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    paper  = get_paper_or_404(paper_id, current_user.id, db)
    result = rag_service.ask(
        paper.vectorstore_path, body.question, body.chat_history
    )

    # Persist messages to DB
    db.add(models.ChatMessage(
        paper_id=paper_id, role="user", content=body.question
    ))
    db.add(models.ChatMessage(
        paper_id=paper_id, role="assistant", content=result["answer"]
    ))
    db.commit()

    return result


@router.get("/{paper_id}/summary", response_model=schemas.SummaryResponse)
def get_summary(
    paper_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    paper   = get_paper_or_404(paper_id, current_user.id, db)
    summary = rag_service.summarize(paper.vectorstore_path)
    return {"summary": summary}


@router.get("/{paper_id}/insights", response_model=schemas.InsightsOut)
def get_insights(
    paper_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    paper = get_paper_or_404(paper_id, current_user.id, db)
    return schemas.InsightsOut(
        title         = paper.title,
        authors       = paper.authors,
        year          = paper.year,
        problem       = paper.problem,
        methodology   = paper.methodology,
        dataset       = paper.dataset,
        contributions = paper.contributions,
        results       = paper.results,
        limitations   = paper.limitations,
    )


@router.get("/{paper_id}/history")
def get_chat_history(
    paper_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    get_paper_or_404(paper_id, current_user.id, db)
    messages = db.query(models.ChatMessage)\
                  .filter(models.ChatMessage.paper_id == paper_id)\
                  .order_by(models.ChatMessage.created_at)\
                  .all()
    return [{"role": m.role, "content": m.content} for m in messages]
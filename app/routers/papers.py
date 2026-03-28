import os
import shutil
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth import get_current_user
from app import models, schemas
from app.services import rag_service, insights_service

router = APIRouter(prefix="/papers", tags=["Papers"])

UPLOAD_DIR      = "data"
VECTORSTORE_DIR = "vectorstores"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)


@router.post("/upload", response_model=schemas.PaperOut)
async def upload_paper(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    # Save PDF
    safe_name       = f"{current_user.id}_{file.filename}"
    file_path       = os.path.join(UPLOAD_DIR, safe_name)
    vectorstore_path = os.path.join(VECTORSTORE_DIR, safe_name.replace(".pdf", ""))

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Process PDF → FAISS
    chunk_count = rag_service.process_pdf(file_path, vectorstore_path)

    # Extract insights
    insights = insights_service.extract_insights(vectorstore_path)

    # Save to DB
    paper = models.Paper(
        user_id          = current_user.id,
        filename         = safe_name,
        original_name    = file.filename,
        file_path        = file_path,
        vectorstore_path = vectorstore_path,
        chunk_count      = chunk_count,
        **{k: insights.get(k) for k in [
            "title","authors","year","problem",
            "methodology","dataset","contributions","results","limitations"
        ]}
    )
    db.add(paper)
    db.commit()
    db.refresh(paper)
    return paper


@router.get("/", response_model=list[schemas.PaperOut])
def get_papers(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    return db.query(models.Paper)\
             .filter(models.Paper.user_id == current_user.id)\
             .order_by(models.Paper.uploaded_at.desc())\
             .all()


@router.delete("/{paper_id}")
def delete_paper(
    paper_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    paper = db.query(models.Paper)\
               .filter(models.Paper.id == paper_id,
                       models.Paper.user_id == current_user.id)\
               .first()
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    # Clean up files
    if os.path.exists(paper.file_path):
        os.remove(paper.file_path)
    if os.path.exists(paper.vectorstore_path):
        shutil.rmtree(paper.vectorstore_path)

    db.delete(paper)
    db.commit()
    return {"message": "Paper deleted"}
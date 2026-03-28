from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage

from app.database import get_db
from app.auth import get_current_user
from app import models, schemas
from app.services.rag_service import load_vectorstore, rerank, get_llm

router = APIRouter(prefix="/compare", tags=["Compare"])


@router.post("/")
def compare_papers(
    body: schemas.CompareRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    def get_paper(pid):
        p = db.query(models.Paper)\
               .filter(models.Paper.id == pid,
                       models.Paper.user_id == current_user.id)\
               .first()
        if not p:
            raise HTTPException(status_code=404, detail=f"Paper {pid} not found")
        return p

    p1 = get_paper(body.paper_id_1)
    p2 = get_paper(body.paper_id_2)

    query = f"What is the {body.aspect} of this paper?"

    def get_context(paper):
        vs   = load_vectorstore(paper.vectorstore_path)
        docs = rerank(query, vs.similarity_search(query, k=8), top_n=3)
        return "\n".join([d.page_content for d in docs])

    ctx1 = get_context(p1)
    ctx2 = get_context(p2)

    prompt = f"""You are an AI research assistant comparing two research papers.

Compare these two papers specifically on: {body.aspect}

Paper 1 — "{p1.title or p1.original_name}":
{ctx1}

Paper 2 — "{p2.title or p2.original_name}":
{ctx2}

Provide a clear structured comparison covering similarities and differences."""

    response = get_llm().invoke([HumanMessage(content=prompt)])

    return {
        "paper_1": p1.title or p1.original_name,
        "paper_2": p2.title or p2.original_name,
        "aspect":  body.aspect,
        "comparison": response.content
    }
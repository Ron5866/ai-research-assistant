import json
from langchain_core.messages import HumanMessage
from app.services.rag_service import load_vectorstore, get_llm


def extract_insights(vectorstore_path: str) -> dict:
    vs   = load_vectorstore(vectorstore_path)
    docs = vs.similarity_search(
        "title authors abstract methodology dataset contributions results", k=6
    )
    context = " ".join([doc.page_content for doc in docs])

    prompt = f"""You are an AI research assistant. Extract structured insights from this research paper.

Context:
{context}

Return ONLY a valid JSON object with exactly these keys:
{{
  "title": "full paper title",
  "authors": "author names as a string",
  "year": "publication year or Unknown",
  "problem": "what problem does this paper solve (1-2 sentences)",
  "methodology": "what approach or method did they use (1-2 sentences)",
  "dataset": "what dataset was used or None",
  "contributions": "top 3 key contributions as a single string",
  "results": "main results or findings (1-2 sentences)",
  "limitations": "stated limitations or None"
}}

Return only the JSON. No explanation, no markdown, no code blocks."""

    response = get_llm().invoke([HumanMessage(content=prompt)])

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return {}
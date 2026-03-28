import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from sentence_transformers import CrossEncoder
from app.config import settings

# Singletons — loaded once
_embeddings = None
_reranker   = None
_llm        = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings()
    return _embeddings

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=settings.GROQ_API_KEY,
            temperature=0
        )
    return _llm


def process_pdf(file_path: str, vectorstore_path: str) -> int:
    """Load PDF, chunk, embed, save FAISS index. Returns chunk count."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    vs = FAISS.from_documents(chunks, get_embeddings())
    vs.save_local(vectorstore_path)

    return len(chunks)


def load_vectorstore(vectorstore_path: str) -> FAISS:
    return FAISS.load_local(
        vectorstore_path,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )


def rerank(query: str, docs: list, top_n: int = 4) -> list:
    if not docs:
        return docs
    reranker = get_reranker()
    pairs  = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]


def ask(vectorstore_path: str, question: str, chat_history: list) -> dict:
    vs       = load_vectorstore(vectorstore_path)
    raw_docs = vs.similarity_search(question, k=10)
    docs     = rerank(question, raw_docs, top_n=4)
    context  = "\n\n".join([doc.page_content for doc in docs])

    history_text = ""
    for msg in chat_history[-6:]:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        history_text += f"{role.capitalize()}: {content}\n"

    prompt = f"""You are an AI research assistant.

Previous conversation:
{history_text}

Use ONLY the context below to answer the question.
Context:
{context}

Question: {question}
Answer clearly:"""

    response = get_llm().invoke([HumanMessage(content=prompt)])

    sources = [
        f"Page {doc.metadata.get('page', '?')}: {doc.page_content[:200]}..."
        for doc in docs
    ]

    return {"answer": response.content, "sources": sources}


def summarize(vectorstore_path: str) -> str:
    vs       = load_vectorstore(vectorstore_path)
    raw_docs = vs.similarity_search("summarize the paper", k=10)
    docs     = rerank("summarize the paper", raw_docs, top_n=5)
    context  = " ".join([doc.page_content for doc in docs])

    prompt = f"""You are an AI research assistant.
Summarize this research paper clearly. Cover: problem, methodology, key findings, contributions.

{context}"""

    response = get_llm().invoke([HumanMessage(content=prompt)])
    return response.content
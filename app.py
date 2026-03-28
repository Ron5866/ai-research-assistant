import streamlit as st
import os
import json
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import CrossEncoder


# ---------------- Setup ----------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# Cross-encoder re-ranker — loads once, reused every query
@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

reranker = load_reranker()


# ---------------- Helpers ----------------

def rerank_docs(query, docs, top_n=3):
    """Re-rank FAISS results using cross-encoder for better relevance."""
    if not docs:
        return docs
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]


def extract_insights(vectorstore, llm):
    """Extract structured metadata from the paper."""
    docs = vectorstore.similarity_search(
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

    response = llm.invoke([HumanMessage(content=prompt)])

    # Safely parse JSON
    try:
        raw = response.content.strip()
        # Strip markdown fences if model adds them anyway
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception:
        return {"error": "Could not parse insights. Try again."}


# ---------------- UI ----------------

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("📄 AI Research Assistant")
st.write("Upload a research paper and ask questions about it.")


# ---------------- Session State ----------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "insights" not in st.session_state:
    st.session_state.insights = None


# ---------------- Sidebar — Paper Insights ----------------

with st.sidebar:
    st.header("📊 Paper Insights")

    if st.session_state.insights:
        ins = st.session_state.insights

        if "error" in ins:
            st.error(ins["error"])
        else:
            st.markdown(f"**📌 Title**\n\n{ins.get('title', 'N/A')}")
            st.markdown(f"**👥 Authors**\n\n{ins.get('authors', 'N/A')}")
            st.markdown(f"**📅 Year**\n\n{ins.get('year', 'N/A')}")
            st.divider()
            st.markdown(f"**🎯 Problem**\n\n{ins.get('problem', 'N/A')}")
            st.markdown(f"**⚙️ Methodology**\n\n{ins.get('methodology', 'N/A')}")
            st.markdown(f"**🗃️ Dataset**\n\n{ins.get('dataset', 'N/A')}")
            st.divider()
            st.markdown(f"**🏆 Contributions**\n\n{ins.get('contributions', 'N/A')}")
            st.markdown(f"**📈 Results**\n\n{ins.get('results', 'N/A')}")
            st.markdown(f"**⚠️ Limitations**\n\n{ins.get('limitations', 'N/A')}")
    else:
        st.info("Upload a paper to see insights here.")


# ---------------- PDF Upload ----------------

uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type="pdf")

if uploaded_file is not None and st.session_state.vectorstore is None:

    if not os.path.exists("data"):
        os.makedirs("data")

    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully")

    with st.spinner("Processing document..."):
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings()
        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)

    st.write(f"Document split into {len(chunks)} chunks")

    # Auto-extract insights on upload
    with st.spinner("Extracting paper insights..."):
        st.session_state.insights = extract_insights(
            st.session_state.vectorstore, llm
        )
    st.success("Insights extracted! Check the sidebar →")


# ---------------- Main Area ----------------

if st.session_state.vectorstore is not None:

    # Summarize button
    if st.button("📑 Summarize Paper"):
        with st.spinner("Generating summary..."):
            raw_docs = st.session_state.vectorstore.similarity_search(
                "summarize the paper", k=10
            )
            # Re-rank before summarizing
            docs = rerank_docs("summarize the paper", raw_docs, top_n=5)
            context = " ".join([doc.page_content for doc in docs])

            prompt = f"""You are an AI research assistant.
Summarize this research paper clearly in simple terms.
Cover: problem, methodology, key findings, and contributions.

{context}"""

            response = llm.invoke([HumanMessage(content=prompt)])
            summary = response.content

        st.subheader("Paper Summary")
        st.write(summary)

        st.session_state.chat_history.append(
            HumanMessage(content="Please summarize this paper.")
        )
        st.session_state.chat_history.append(AIMessage(content=summary))

    st.divider()

    # ---------------- Chat ----------------

    st.subheader("💬 Chat with the Paper")

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

    question = st.chat_input("Ask a question about the paper...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Thinking..."):

            # Step 1 — broad FAISS retrieval
            raw_docs = st.session_state.vectorstore.similarity_search(
                question, k=10
            )

            # Step 2 — re-rank with cross-encoder
            reranked_docs = rerank_docs(question, raw_docs, top_n=4)

            # Step 3 — build context from re-ranked docs
            context = "\n\n".join([doc.page_content for doc in reranked_docs])

            # Step 4 — build prompt with memory
            history_text = ""
            for msg in st.session_state.chat_history[-6:]:
                if isinstance(msg, HumanMessage):
                    history_text += f"User: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    history_text += f"Assistant: {msg.content}\n"

            prompt = f"""You are an AI research assistant.

Previous conversation:
{history_text}

Use ONLY the context below to answer the question.
Context:
{context}

Question: {question}

Answer clearly and cite which part of the paper supports your answer:"""

            response = llm.invoke([HumanMessage(content=prompt)])
            answer = response.content

        with st.chat_message("assistant"):
            st.write(answer)

            # Show source chunks used
            with st.expander("📎 Source chunks used"):
                for i, doc in enumerate(reranked_docs):
                    st.markdown(f"**Chunk {i+1}** (page {doc.metadata.get('page', '?')})")
                    st.caption(doc.page_content[:300] + "...")

        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=answer))

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

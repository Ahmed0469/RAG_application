"""Single-file Streamlit dashboard for session-isolated document RAG."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".csv", ".txt", ".md"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
GROQ_MODEL_NAME = "openai/gpt-oss-20b"

# These objects belong only to one Streamlit session. The embedding model is
# intentionally excluded because it is the one shared, server-wide resource.
DOCUMENT_STATE_KEYS = (
    "document_name",
    "document_bytes",
    "document_fingerprint",
    "documents",
    "chunks",
    "nodes",
    "qdrant_client",
    "vector_store",
    "index",
    "vector_retriever",
    "bm25_retriever",
    "hybrid_retriever",
    "query_engine",
    "messages",
    "index_error",
    "attempted_fingerprint",
    "pending_question",
)


@st.cache_resource(show_spinner="Loading the shared BGE-M3 embedding model...")
def get_embedding_model() -> HuggingFaceEmbedding:
    """Load one reusable embedding model for all users of this server."""
    return HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)


def reset_document_state() -> None:
    """Release all session-owned document resources and clear its chat."""
    client = st.session_state.get("qdrant_client")
    if client is not None:
        try:
            client.close()
        except Exception:
            logger.exception("Could not close the session Qdrant client")

    for key in DOCUMENT_STATE_KEYS:
        st.session_state.pop(key, None)


def get_upload_fingerprint(uploaded_file: Any) -> str | None:
    """Identify a file by both its name and bytes, without persisting it."""
    if uploaded_file is None:
        return None
    digest = hashlib.sha256()
    digest.update(uploaded_file.name.encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(uploaded_file.getvalue())
    return digest.hexdigest()


def on_upload_change(widget_key: str = "document_upload") -> None:
    """Select either uploader's value and clear prior document resources."""
    uploaded_file = st.session_state.get(widget_key)
    new_fingerprint = get_upload_fingerprint(uploaded_file)
    old_fingerprint = st.session_state.get("upload_fingerprint")

    if new_fingerprint != old_fingerprint:
        reset_document_state()
        st.session_state["upload_fingerprint"] = new_fingerprint
    st.session_state["selected_upload"] = uploaded_file
    st.session_state["upload_selection_set"] = True


def add_other_document() -> None:
    """Clear the active document and reset both uploader widgets."""
    reset_document_state()
    for key in ("document_upload", "welcome_upload"):
        st.session_state.pop(key, None)
    st.session_state["upload_fingerprint"] = None
    st.session_state["selected_upload"] = None
    st.session_state["upload_selection_set"] = False


def get_groq_api_key() -> str | None:
    """Prefer a local environment key, with Streamlit secrets as a fallback."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if api_key:
        return api_key
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        return None
    return str(api_key).strip() or None


def load_uploaded_documents(filename: str, content: bytes) -> list[Any]:
    """Use the same LangChain loaders as the former upload endpoint."""
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Upload a PDF, CSV, TXT, or Markdown file.")

    # Existing LangChain loaders accept paths, so use a temporary file and
    # remove it immediately after parsing; no uploaded file is persisted.
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp:
            temp.write(content)
            temporary_path = temp.name

        if extension == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader

            return PyPDFLoader(temporary_path).load()
        if extension == ".csv":
            from langchain_community.document_loaders import CSVLoader

            try:
                return CSVLoader(temporary_path, encoding="utf-8-sig").load()
            except RuntimeError as error:
                if not isinstance(error.__cause__, UnicodeDecodeError):
                    raise
                return CSVLoader(temporary_path, encoding="latin-1").load()

        from langchain_community.document_loaders import TextLoader

        return TextLoader(temporary_path).load()
    finally:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)


def build_document_resources(
    filename: str, content: bytes, api_key: str
) -> dict[str, Any]:
    """Create this session's chunks, in-memory index, retrievers, and engine."""
    documents = load_uploaded_documents(filename, content)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    ).split_documents(documents)
    if not chunks:
        raise ValueError("No readable text was found in this document.")

    # Keep loader metadata on each node so PDF page labels survive retrieval.
    nodes = [
        TextNode(
            id_=str(uuid.uuid4()),
            text=chunk.page_content,
            metadata=chunk.metadata,
        )
        for chunk in chunks
    ]

    # Each user receives a private in-memory Qdrant client and collection.
    qdrant_client = QdrantClient(":memory:")
    try:
        vector_store = QdrantVectorStore(
            collection_name=f"document_{uuid.uuid4().hex}",
            client=qdrant_client,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=get_embedding_model(),
            show_progress=False,
        )

        vector_retriever = index.as_retriever(similarity_top_k=5)
        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
        hybrid_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=5,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
            llm=MockLLM(),
            verbose=False,
        )
        llm = Groq(model=GROQ_MODEL_NAME, temperature=0.2, api_key=api_key)
        query_engine = RetrieverQueryEngine.from_args(
            retriever=hybrid_retriever,
            llm=llm,
        )

        return {
            "documents": documents,
            "chunks": chunks,
            "nodes": nodes,
            "qdrant_client": qdrant_client,
            "vector_store": vector_store,
            "index": index,
            "vector_retriever": vector_retriever,
            "bm25_retriever": bm25_retriever,
            "hybrid_retriever": hybrid_retriever,
            "query_engine": query_engine,
        }
    except Exception:
        qdrant_client.close()
        raise


def format_sources(response: Any, document_name: str) -> list[dict[str, str]]:
    """Extract concise source references from a LlamaIndex response."""
    sources = []
    for source in getattr(response, "source_nodes", []):
        metadata = source.node.metadata
        page = metadata.get("page_label") or metadata.get("page")
        sources.append(
            {
                "label": f"Page {page}" if page is not None else document_name,
                "excerpt": source.node.text[:280],
            }
        )
    return sources


def render_chat_message(message: dict[str, Any]) -> None:
    """Render a stored conversation turn with its optional citations."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for number, source in enumerate(message.get("sources", []), start=1):
            with st.expander(f"Source {number:02d} · {source['label']}"):
                st.caption(source["excerpt"])


def queue_question(question: str) -> None:
        """Queue a suggested prompt for the next app run."""
        st.session_state["pending_question"] = question


def clear_chat() -> None:
        """Start a fresh conversation without discarding the indexed document."""
        st.session_state["messages"] = []
        st.session_state.pop("pending_question", None)


def inject_theme() -> None:
        """Apply the reference site's editorial black, ivory, and green theme."""
        st.markdown(
                """
                <style>
                @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Instrument+Serif:ital@0;1&display=swap');
                :root {
                    color-scheme: dark;
                    --ink: #090908;
                    --panel: #10100e;
                    --line: rgba(235, 228, 207, .13);
                    --muted: #85847e;
                    --paper: #eee8d9;
                    --green: #38d889;
                    --mono: 'DM Mono', monospace;
                    --sans: 'DM Sans', sans-serif;
                    --serif: 'Instrument Serif', Georgia, serif;
                }
                html, body, [class*="stApp"] {
                    background: var(--ink);
                    color: var(--paper);
                    font-family: var(--sans);
                }
                .stApp {
                    background-image:
                        radial-gradient(ellipse at 76% 4%, rgba(68, 67, 48, .19), transparent 33%),
                        linear-gradient(rgba(235,228,207,.025) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(235,228,207,.025) 1px, transparent 1px);
                    background-size: auto, 54px 54px, 54px 54px;
                }
                [data-testid="stAppViewContainer"] > .main { background: transparent; }
                .block-container { max-width: 1360px; padding-top: 1.2rem; padding-bottom: 4rem; }
                header[data-testid="stHeader"] { background: transparent; }
                [data-testid="stToolbar"] { right: 1rem; }
                section[data-testid="stSidebar"] {
                    background: rgba(12, 12, 11, .96);
                    border-right: 1px solid var(--line);
                }
                section[data-testid="stSidebar"] > div { padding-top: 1.4rem; }
                .topbar {
                    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: .6rem;
                    min-height: 62px; margin-bottom: 2.2rem; padding: 2px 8px 4px; border-bottom: 1px solid var(--line);
                }
                .brand { display: inline-flex; flex-shrink: 0; align-items: center; transform: translateY(8px); color: var(--paper); font: 22px/1.4 var(--serif); white-space: nowrap; }
                .brand span { color: var(--green); }
                .top-links, .availability, .eyebrow, .micro, .metric-label, .section-tag {
                    color: var(--muted); font: 10px var(--mono); letter-spacing: 1.1px; text-transform: uppercase;
                }
                .top-links { display: none; }
                .top-links a { color: inherit; text-decoration: none; }
                .availability { display: flex; flex-shrink: 0; align-items: center; gap: 8px; font-size: 9px; letter-spacing: .5px; white-space: nowrap; }
                .live-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); box-shadow: 0 0 12px rgba(56,216,137,.55); }
                .hero { display: block; margin: 2rem 0 2.3rem; }
                .eyebrow { margin-bottom: 1rem; }
                .eyebrow::before { content: '> '; color: var(--green); }
                .hero h1 { margin: 0; color: var(--paper); font: 400 clamp(46px, 5.2vw, 68px)/.96 var(--serif); letter-spacing: 0; }
                .hero h1 em { color: #b4a67e; font-weight: 400; }
                .hero-copy { max-width: 620px; margin: 1rem 0 0; color: #98968e; font: 14px/1.8 var(--sans); }
                .metric-row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); margin: 0 0 2.6rem; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
                .metric { padding: 15px 18px 15px 0; }
                .metric + .metric { padding-left: 18px; border-left: 1px solid var(--line); }
                .metric-value { color: var(--paper); font: 28px/1 var(--serif); }
                .metric-label { margin-top: 7px; font-size: 9px; }
                .workspace-head { display: flex; justify-content: space-between; align-items: center; gap: 1rem; margin: 0 0 1rem; padding-bottom: 12px; border-bottom: 1px solid var(--line); }
                .workspace-title { color: var(--paper); font: 24px var(--serif); }
                .workspace-state { color: #aaa797; font: 9px var(--mono); letter-spacing: .7px; text-transform: uppercase; }
                .sidebar-brand { margin: .4rem 0 1.7rem; color: var(--paper); font: 25px var(--serif); }
                .sidebar-brand small { display: block; margin-top: 3px; color: var(--muted); font: 9px var(--mono); letter-spacing: 1.2px; text-transform: uppercase; }
                .sidebar-note { padding: 12px; border: 1px solid var(--line); border-radius: 3px; color: #aaa79c; background: rgba(255,255,255,.018); font-size: 11px; line-height: 1.6; }
                .section-tag { margin: 1.3rem 0 .7rem; color: #aaa58f; }
                [data-testid="stFileUploader"] section {
                    border: 1px dashed rgba(238,232,217,.27); border-radius: 4px; background: rgba(255,255,255,.018);
                }
                [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] small { color: #c6c1b5 !important; }
                [data-testid="stFileUploader"] button, .stButton > button, .stDownloadButton > button {
                    min-height: 40px; border: 1px solid rgba(238,232,217,.24); border-radius: 3px;
                    color: var(--paper); background: rgba(255,255,255,.025); font: 11px var(--mono);
                    transition: background .18s ease, border-color .18s ease, transform .18s ease;
                }
                [data-testid="stFileUploader"] button { color: #07110b !important; border-color: var(--green) !important; background: var(--green) !important; }
                [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] p { color: #c6c1b5 !important; }
                [data-testid="stFileUploader"] button span, [data-testid="stFileUploader"] button p { color: #07110b !important; }
                .st-key-welcome_upload_cta {
                    position: relative; display: flex; min-height: 245px; align-items: center; justify-content: center;
                    overflow: hidden; padding: 1.5rem !important; border: 1px solid var(--line) !important; border-radius: 2px;
                    background: rgba(255,255,255,.012) !important; text-align: center; cursor: pointer;
                    transition: border-color .2s ease, background .2s ease, box-shadow .2s ease;
                }
                .stApp [data-testid="stMain"] div.st-key-welcome_upload_cta:hover,
                .stApp [data-testid="stMain"] div.st-key-welcome_upload_cta:focus-within {
                    border-color: var(--green) !important; background: rgba(56,216,137,.055) !important;
                    box-shadow: inset 0 0 0 1px rgba(56,216,137,.2) !important;
                }
                .st-key-welcome_upload_cta [data-testid="stMarkdownContainer"] { position: relative; z-index: 1; pointer-events: none; }
                .st-key-welcome_upload_cta .empty-mark { transition: color .2s ease, border-color .2s ease, transform .2s ease; }
                .stApp [data-testid="stMain"] div.st-key-welcome_upload_cta:hover .empty-mark,
                .stApp [data-testid="stMain"] div.st-key-welcome_upload_cta:focus-within .empty-mark {
                    transform: translateY(-2px); border-color: var(--green) !important; color: var(--green) !important;
                }
                .st-key-welcome_upload_cta .empty-content { width: 100%; margin: 0; padding: 0; border: 0; background: transparent; text-align: center; }
                .st-key-welcome_upload_cta .st-key-welcome_upload { position: static !important; height: 0 !important; min-height: 0 !important; margin: 0 !important; }
                .st-key-welcome_upload_cta [data-testid="stFileUploader"] { position: static; height: 0; margin: 0; }
                .st-key-welcome_upload_cta [data-testid="stFileUploaderDropzone"] {
                    position: absolute; z-index: 2; inset: 0; display: block; padding: 0 !important;
                    border: 0 !important; border-radius: 0 !important; background: transparent !important; cursor: pointer;
                }
                .st-key-welcome_upload_cta [data-testid="stFileUploaderDropzone"] button {
                    position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0; cursor: pointer;
                }
                .st-key-welcome_upload_cta [data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
                .st-key-welcome_upload_cta [data-testid="stFileChip"],
                .st-key-welcome_upload_cta [data-testid="stBaseButton-borderlessIcon"] { display: none !important; }
                .stButton > button:hover, .stDownloadButton > button:hover {
                    color: var(--paper); border-color: var(--green); background: rgba(56,216,137,.08); transform: translateY(-1px);
                }
                .stButton > button[kind="primary"] { color: #07110b; border-color: var(--green); background: var(--green); }
                [data-testid="stChatMessage"] { color: var(--paper) !important; border: 1px solid var(--line); border-radius: 4px; background: rgba(16,16,14,.94); }
                [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] span { color: var(--paper) !important; font-size: 14px; line-height: 1.75; }
                [data-testid="stBottom"] { background: linear-gradient(transparent, var(--ink) 22%) !important; }
                [data-testid="stBottom"] > div { background: var(--ink) !important; }
                [data-testid="stChatInput"] { background: transparent !important; }
                [data-testid="stChatInput"] > div { border: 1px solid rgba(238,232,217,.3) !important; border-radius: 5px !important; background: #151511 !important; box-shadow: 0 8px 28px rgba(0,0,0,.34) !important; }
                [data-testid="stChatInput"] textarea { border: 0 !important; border-radius: 4px !important; outline: 0 !important; background: transparent !important; color: #eee8d9 !important; -webkit-text-fill-color: #eee8d9 !important; caret-color: var(--green) !important; }
                [data-testid="stChatInput"] textarea::placeholder { color: #eee8d9 !important; -webkit-text-fill-color: #eee8d9 !important; opacity: 1 !important; }
                [data-testid="stChatInput"] button { color: #07110b !important; border: 0 !important; background: var(--green) !important; }
                [data-testid="stChatInput"] button:disabled { color: #07110b !important; background: #6d8f7c !important; opacity: .85 !important; }
                [data-testid="stExpander"] { color: var(--paper) !important; border: 1px solid var(--line); border-radius: 3px; background: rgba(255,255,255,.018); }
                [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p, [data-testid="stExpander"] details, [data-testid="stExpander"] p { color: var(--paper) !important; }
                [data-testid="stStatusWidget"] { color: #eee8d9 !important; border: 1px solid rgba(238,232,217,.2) !important; border-radius: 4px; background: #10100e !important; }
                [data-testid="stStatusWidget"] *, [data-testid="stStatusWidget"] p { color: #eee8d9 !important; }
                [data-testid="stStatusWidget"] details,
                [data-testid="stStatusWidget"] summary,
                [data-testid="stStatusWidget"] summary:hover,
                [data-testid="stStatusWidget"] details > div { color: #eee8d9 !important; background: #10100e !important; }
                [data-testid="stStatusWidget"] summary { border-radius: 4px 4px 0 0; }
                [data-testid="stStatusWidget"] svg { color: var(--green) !important; }
                .stApp [data-testid="stMain"] details,
                .stApp [data-testid="stMain"] details > summary,
                .stApp [data-testid="stMain"] details > summary:hover,
                .stApp [data-testid="stMain"] details > summary * { color: #eee8d9 !important; background-color: #10100e !important; }
                [data-testid="stAlert"] { color: #eee8d9 !important; }
                [data-testid="stAlert"] p, [data-testid="stAlert"] span { color: #eee8d9 !important; }
                [data-testid="stAlert"] { border-radius: 3px; }
                .empty-state { margin: 1rem 0 1.4rem; padding: 2.5rem 1rem; border: 1px solid var(--line); text-align: center; background: rgba(255,255,255,.012); }
                .empty-mark { width: 42px; height: 42px; display: grid; place-items: center; margin: 0 auto 14px; border: 1px solid rgba(238,232,217,.25); border-radius: 50%; color: #c5b78e; font: 21px var(--serif); }
                .empty-mark.ready-mark { border-color: rgba(56,216,137,.42); color: var(--green); font: 20px var(--sans); }
                .empty-title { color: var(--paper); font: 25px var(--serif); }
                .empty-copy { margin: 7px auto 0; color: var(--muted); font-size: 12px; }
                .stCaption, [data-testid="stCaptionContainer"] { color: var(--muted); }
                hr { border-color: var(--line); }
                @media (max-width: 760px) {
                    .block-container { padding-top: .7rem; }
                    .top-links { display: none; }
                    .hero { grid-template-columns: 1fr; gap: 1rem; margin-top: 1.2rem; }
                    .hero h1 { font-size: 58px; }
                    .hero-copy { max-width: 520px; }
                    .metric { padding-right: 7px; }
                    .metric + .metric { padding-left: 9px; }
                    .metric-value { font-size: 24px; }
                    .metric-label { font-size: 8px; letter-spacing: .4px; }
                    .workspace-head { align-items: flex-start; flex-direction: column; gap: 5px; }
                }
                @media (prefers-reduced-motion: reduce) { *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; } }
                </style>
                """,
                unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Docstore | Ask your documents",
        page_icon="D",
        layout="wide",
    )

    inject_theme()

    st.markdown(
        """
                <div class="topbar">
                    <div class="brand"><span>+</span> Docstore</div>
                    <nav class="top-links"><a href="#workspace">Workspace</a><a href="#sources">Sources</a></nav>
                    <div class="availability"><span class="live-dot"></span> PRIVATE SESSION</div>
                </div>
                <section class="hero">
                    <div>
                        <div class="eyebrow">Local document intelligence</div>
                        <h1>Answers hiding<br>in your <em>documents.</em></h1>
                    </div>
                    <p class="hero-copy">Bring a source into focus. Ask naturally, get grounded answers, and trace every response back to the page it came from.</p>
                </section>
            """,
            unsafe_allow_html=True,
            )

    # Ensure state keys exist before widgets and callbacks can inspect them.
    st.session_state.setdefault("upload_fingerprint", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("selected_upload", None)
    st.session_state.setdefault("upload_selection_set", False)
    upload_is_locked = st.session_state["selected_upload"] is not None

    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand">Docstore<small>RAG / Workspace 01</small></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-tag" id="sources">01 / SOURCE FILE</div>',
            unsafe_allow_html=True,
        )
        sidebar_upload = st.file_uploader(
            "Add or replace a document",
            type=["pdf", "csv", "txt", "md"],
            accept_multiple_files=False,
            max_upload_size=50,
            key="document_upload",
            on_change=on_upload_change,
            disabled=upload_is_locked,
        )
        st.caption("PDF · CSV · TXT · MD  /  MAX 50 MB")

        if st.session_state["selected_upload"] is not None and sidebar_upload is None:
            selected_upload = st.session_state["selected_upload"]
            st.markdown('<div class="section-tag">CURRENT SOURCE</div>', unsafe_allow_html=True)
            st.write(selected_upload.name)
            st.caption(f"{selected_upload.size / (1024 * 1024):.1f} MB · selected for this session")

        if st.session_state.get("query_engine") is not None:
            st.success(f"Source indexed · {st.session_state['document_name']}")
            st.caption(f"{len(st.session_state['chunks'])} chunks ready")
            st.markdown('<div class="section-tag">02 / RETRIEVAL PIPELINE</div>', unsafe_allow_html=True)
            st.markdown("`BGE-M3`  →  `QDRANT`  +  `BM25`  →  `GROQ`")
            st.caption("Reciprocal-rank fusion · Top 5 sources")
        elif st.session_state.get("index_error"):
            st.error(st.session_state["index_error"])
            if st.button("Retry indexing", use_container_width=True):
                st.session_state.pop("attempted_fingerprint", None)
                st.rerun()
        else:
            st.markdown(
                '<div class="sidebar-note">Upload a source to start a private session.</div>',
                unsafe_allow_html=True,
            )

        if upload_is_locked:
            st.button(
                "Add other document",
                on_click=add_other_document,
                use_container_width=True,
                type="primary",
            )

        st.divider()
        st.caption("Add another source to clear this document, its index, and chat.")

    welcome_upload = None
    if st.session_state.get("query_engine") is None:
        with st.container(key="welcome_upload_cta"):
            st.markdown(
                '<div class="empty-content"><div class="empty-mark">+</div><div class="empty-title">Your document, ready to explore.</div><div class="empty-copy">Choose a source to build a private hybrid search index.</div></div>',
                unsafe_allow_html=True,
            )
            welcome_upload = st.file_uploader(
                "Start with a source",
                type=["pdf", "csv", "txt", "md"],
                accept_multiple_files=False,
                max_upload_size=50,
                key="welcome_upload",
                on_change=on_upload_change,
                args=("welcome_upload",),
                label_visibility="collapsed",
                disabled=upload_is_locked,
            )

    if not st.session_state["upload_selection_set"]:
        restored_upload = sidebar_upload or welcome_upload
        if restored_upload is not None:
            st.session_state["selected_upload"] = restored_upload
            st.session_state["upload_selection_set"] = True

    uploaded_file = st.session_state["selected_upload"]

    # A defensive fingerprint check also covers a changed uploader value if
    # the session callback was interrupted before it could reset the state.
    current_fingerprint = get_upload_fingerprint(uploaded_file)
    if uploaded_file is None and st.session_state.get("query_engine") is not None:
        current_fingerprint = st.session_state.get("document_fingerprint")
    elif current_fingerprint != st.session_state.get("upload_fingerprint"):
        reset_document_state()
        st.session_state["upload_fingerprint"] = current_fingerprint
        st.session_state["selected_upload"] = uploaded_file

    if uploaded_file is not None:
        file_content = uploaded_file.getvalue()
        if len(file_content) > MAX_UPLOAD_BYTES:
            reset_document_state()
            st.session_state["upload_fingerprint"] = current_fingerprint
            st.error("The maximum upload size is 50 MB.")
        elif not file_content:
            st.error("The uploaded file is empty.")
        else:
            st.session_state["document_name"] = uploaded_file.name
            st.session_state["document_bytes"] = file_content

            if (
                st.session_state.get("document_fingerprint") != current_fingerprint
                and st.session_state.get("attempted_fingerprint")
                != current_fingerprint
            ):
                api_key = get_groq_api_key()
                if not api_key:
                    st.session_state["attempted_fingerprint"] = current_fingerprint
                    error_message = (
                        "Set GROQ_API_KEY in your environment (or Streamlit secrets) to index a document."
                    )
                    st.session_state["index_error"] = error_message
                    st.error(error_message)
                    st.rerun()
                else:
                    st.session_state["attempted_fingerprint"] = current_fingerprint
                    try:
                        with st.status("Preparing your document...", expanded=True) as status:
                            st.write("Parsing the source and building vector + keyword indexes")
                            resources = build_document_resources(
                                uploaded_file.name,
                                file_content,
                                api_key,
                            )
                            for key, value in resources.items():
                                st.session_state[key] = value
                            st.session_state["document_fingerprint"] = current_fingerprint
                            st.session_state["index_error"] = None
                            status.update(label="Hybrid index ready", state="complete")
                        st.rerun()
                    except Exception:
                        logger.exception("Could not index uploaded document")
                        reset_document_state()
                        st.session_state["upload_fingerprint"] = current_fingerprint
                        st.session_state["attempted_fingerprint"] = current_fingerprint
                        st.session_state["document_name"] = uploaded_file.name
                        st.session_state["document_bytes"] = file_content
                        error_message = (
                            "Indexing failed. Check the server logs, then retry."
                        )
                        st.session_state["index_error"] = error_message
                        st.error(error_message)
                        st.rerun()

    document_ready = st.session_state.get("query_engine") is not None
    chunk_count = len(st.session_state.get("chunks", []))
    metric_columns = st.columns(3)
    metrics = (
        ("01" if document_ready else "00", "ACTIVE SOURCE"),
        (f"{chunk_count:02d}", "SEARCHABLE CHUNKS"),
        ("HYBRID", "RETRIEVAL MODE"),
    )
    for column, (value, label) in zip(metric_columns, metrics):
        with column:
            st.markdown(
                f'<div class="metric"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div id="workspace"></div>', unsafe_allow_html=True)
    header_columns = st.columns([0.72, 0.28])
    with header_columns[0]:
        st.markdown(
            '<div class="workspace-head"><span class="workspace-title">Conversation</span><span class="workspace-state">Source-grounded / Session private</span></div>',
            unsafe_allow_html=True,
        )
    with header_columns[1]:
        if st.session_state.get("messages"):
            download_text = "\n\n".join(
                f"{message['role'].upper()}\n{message['content']}"
                for message in st.session_state["messages"]
            )
            action_columns = st.columns(2)
            with action_columns[0]:
                st.download_button(
                    "Export chat",
                    data=download_text,
                    file_name="docstore-conversation.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with action_columns[1]:
                st.button("Clear chat", on_click=clear_chat, use_container_width=True)

    if document_ready:
        for stored_message in st.session_state["messages"]:
            render_chat_message(stored_message)

        if not st.session_state["messages"]:
            st.markdown(
                f"""
                <div class="empty-state">
                                    <div class="empty-mark ready-mark">&#10003;</div>
                  <div class="empty-title">Your source is ready.</div>
                  <div class="empty-copy">Ask a question below, or start with a prompt.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            suggestions = (
                "Summarize this document",
                "What are the key requirements?",
                "Find important dates or figures",
            )
            prompt_columns = st.columns(3)
            for column, prompt in zip(prompt_columns, suggestions):
                with column:
                    st.button(
                        prompt,
                        key=f"suggestion_{prompt}",
                        on_click=queue_question,
                        args=(prompt,),
                        use_container_width=True,
                    )
    if document_ready:
        question = st.chat_input(
            "Ask a question about your document...",
            max_chars=4000,
        )
        question = question or st.session_state.pop("pending_question", None)
        if question:
            st.session_state["messages"].append(
                {"role": "user", "content": question, "sources": []}
            )
            render_chat_message(st.session_state["messages"][-1])

            with st.chat_message("assistant"):
                with st.spinner("Searching vector and keyword indexes..."):
                    try:
                        response = st.session_state["query_engine"].query(
                            question.strip()
                        )
                        answer = str(response)
                        sources = format_sources(
                            response,
                            st.session_state["document_name"],
                        )
                    except Exception:
                        logger.exception("RAG query failed")
                        answer = (
                            "The question could not be answered. Check the server "
                            "logs and try again."
                        )
                        sources = []
                st.markdown(answer)
                for number, source in enumerate(sources, start=1):
                    with st.expander(f"Source {number:02d} · {source['label']}"):
                        st.caption(source["excerpt"])

            st.session_state["messages"].append(
                {"role": "assistant", "content": answer, "sources": sources}
            )


if __name__ == "__main__":
    main()
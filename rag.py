from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"


def _pdf_to_documents(pdf_bytes: bytes, source_name: str) -> list[Document]:
    """Extract PDF text while preserving page numbers and source names."""
    documents: list[Document] = []
    reader = PdfReader(BytesIO(pdf_bytes))

    for page_index, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""

        if page_text.strip():
            documents.append(
                Document(
                    page_content=page_text.strip(),
                    metadata={
                        "source": source_name,
                        "page": page_index + 1,
                    },
                )
            )

    return documents


def _load_fixed_pdfs() -> list[Document]:
    """Load PDFs placed by the project owner in knowledge_base/."""
    documents: list[Document] = []
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

    for pdf_path in sorted(KNOWLEDGE_BASE_DIR.glob("*.pdf")):
        try:
            documents.extend(
                _pdf_to_documents(pdf_path.read_bytes(), pdf_path.name)
            )
        except Exception as exc:
            # Keep other documents usable if one PDF is damaged.
            documents.append(
                Document(
                    page_content=f"Could not read PDF {pdf_path.name}: {exc}",
                    metadata={"source": pdf_path.name, "page": 0},
                )
            )

    return documents


def _uploaded_pdfs_to_documents(uploaded_files: Iterable) -> list[Document]:
    documents: list[Document] = []

    for uploaded in uploaded_files or []:
        try:
            data = uploaded.getvalue()
            if data:
                documents.extend(_pdf_to_documents(data, uploaded.name))
        except Exception:
            # An invalid PDF is skipped; the app can still use fixed PDFs.
            continue

    return documents


def build_rag_index(uploaded_files: Iterable = ()):
    """
    Build an in-memory vector index from fixed and currently uploaded PDFs.
    Returns (vector_store, document_count, fingerprint).
    """
    fixed_docs = _load_fixed_pdfs()
    uploaded_docs = _uploaded_pdfs_to_documents(uploaded_files)
    all_docs = fixed_docs + uploaded_docs

    if not all_docs:
        return None, 0, "empty"

    # Exclude unreadable-PDF warning documents from embedding.
    all_docs = [
        doc for doc in all_docs
        if doc.page_content.strip()
        and not doc.page_content.startswith("Could not read PDF ")
    ]

    if not all_docs:
        return None, 0, "empty"

    fingerprint_parts = []
    for doc in all_docs:
        fingerprint_parts.append(
            f"{doc.metadata.get('source')}|{doc.metadata.get('page')}|"
            f"{sha256(doc.page_content.encode('utf-8')).hexdigest()}"
        )
    fingerprint = sha256("\n".join(fingerprint_parts).encode("utf-8")).hexdigest()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        add_start_index=True,
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    return vector_store, len(all_docs), fingerprint


def retrieve_context(vector_store, question: str, k: int = 5) -> tuple[str, list[dict]]:
    """Retrieve relevant PDF passages and return context plus source references."""
    if vector_store is None:
        return "", []

    matches = vector_store.similarity_search(question, k=k)
    context_parts = []
    sources = []

    for index, doc in enumerate(matches, start=1):
        source = str(doc.metadata.get("source", "Unknown document"))
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Source {index}: {source}, page {page}]\n{doc.page_content}"
        )
        sources.append({"source": source, "page": page})

    return "\n\n".join(context_parts), sources

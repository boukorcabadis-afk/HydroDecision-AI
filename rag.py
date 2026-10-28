from __future__ import annotations

import time
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
CHROMA_DIR = Path(__file__).parent / "chroma_db"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
EMBED_BATCH_SIZE = 10
MAX_RETRIES = 5


def _pdf_to_documents(pdf_bytes: bytes, source_name: str) -> list[Document]:
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
    documents: list[Document] = []
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

    for pdf_path in sorted(KNOWLEDGE_BASE_DIR.glob("*.pdf")):
        try:
            documents.extend(
                _pdf_to_documents(pdf_path.read_bytes(), pdf_path.name)
            )
        except Exception as exc:
            print(f"Could not read PDF {pdf_path.name}: {exc}")

    return documents


def _uploaded_pdfs_to_documents(uploaded_files: Iterable) -> list[Document]:
    documents: list[Document] = []

    for uploaded in uploaded_files or []:
        try:
            data = uploaded.getvalue()
            if data:
                documents.extend(_pdf_to_documents(data, uploaded.name))
        except Exception as exc:
            print(f"Could not read uploaded PDF: {exc}")

    return documents


def _fingerprint_documents(documents: list[Document]) -> str:
    parts = []

    for doc in documents:
        parts.append(
            f"{doc.metadata.get('source')}|{doc.metadata.get('page')}|"
            f"{sha256(doc.page_content.encode('utf-8')).hexdigest()}"
        )

    return sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _embed_with_retry(embeddings, texts: list[str]) -> list[list[float]]:
    for attempt in range(MAX_RETRIES):
        try:
            return embeddings.embed_documents(texts)
        except Exception as exc:
            message = str(exc).lower()

            if "429" not in message and "resource_exhausted" not in message:
                raise

            if attempt == MAX_RETRIES - 1:
                raise

            wait_seconds = 60 * (attempt + 1)
            print(
                f"Gemini embedding quota reached. "
                f"Waiting {wait_seconds} seconds before retry."
            )
            time.sleep(wait_seconds)

    raise RuntimeError("Embedding failed after retries.")


def build_rag_index(uploaded_files: Iterable = ()):
    """
    Build a persistent vector index from fixed and uploaded PDFs.
    Returns (vector_store, document_count, fingerprint).
    """
    fixed_docs = _load_fixed_pdfs()
    uploaded_docs = _uploaded_pdfs_to_documents(uploaded_files)
    all_docs = fixed_docs + uploaded_docs

    all_docs = [
        doc for doc in all_docs
        if doc.page_content.strip()
        and not doc.page_content.startswith("Could not read PDF ")
    ]

    if not all_docs:
        return None, 0, "empty"

    fingerprint = _fingerprint_documents(all_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name="hydrodecision_pdf",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    # Avoid duplicating chunks already stored for the same documents.
    existing = vector_store.get(include=["metadatas"])
    existing_ids = set(existing.get("ids", []))

    new_chunks = []
    new_ids = []

    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        page = str(chunk.metadata.get("page", "?"))
        start = str(chunk.metadata.get("start_index", 0))
        content_hash = sha256(
            chunk.page_content.encode("utf-8")
        ).hexdigest()[:16]

        chunk_id = sha256(
            f"{source}|{page}|{start}|{content_hash}".encode("utf-8")
        ).hexdigest()

        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    for start_index in range(0, len(new_chunks), EMBED_BATCH_SIZE):
        batch_docs = new_chunks[start_index:start_index + EMBED_BATCH_SIZE]
        batch_ids = new_ids[start_index:start_index + EMBED_BATCH_SIZE]

        for attempt in range(MAX_RETRIES):
            try:
                vector_store.add_documents(
                    documents=batch_docs,
                    ids=batch_ids,
                )
                break
            except Exception as exc:
                message = str(exc).lower()

                if (
                    "429" not in message
                    and "resource_exhausted" not in message
                ):
                    raise

                if attempt == MAX_RETRIES - 1:
                    raise

                time.sleep(60 * (attempt + 1))

        # Small pause between batches to reduce rate-limit errors.
        if start_index + EMBED_BATCH_SIZE < len(new_chunks):
            time.sleep(2)

    return vector_store, len(all_docs), fingerprint


def retrieve_context(
    vector_store, question: str, k: int = 5
) -> tuple[str, list[dict]]:
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

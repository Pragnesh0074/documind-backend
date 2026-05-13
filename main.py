import math
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

# SQLite3 fix for ChromaDB on Render
try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="DocBot Backend")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("uploads")
CHROMA_DB_DIR = "chroma_db"
COLLECTION_NAME = "pdf_knowledge"
MAX_HISTORY_MESSAGES = 20
RETRIEVAL_CANDIDATES = 28
RERANK_TOP_K = 9
BATCH_SIZE = 40

UPLOAD_DIR.mkdir(exist_ok=True)

# Pre-initialize models to avoid timeouts during upload
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
    streaming=True,
)

# Global state for this lightweight single-user app.
state_lock = threading.RLock()
vectorstore: Optional[Chroma] = None
chat_history: List = []  # stores HumanMessage / AIMessage objects
documents: Dict[str, dict] = {}
upload_jobs: Dict[str, dict] = {}


class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []
    document_ids: Optional[List[str]] = None


def display_name(filename: str) -> str:
    stem = Path(filename).stem.replace("_", " ").replace("-", " ")
    return " ".join(word.capitalize() for word in stem.split()) or "Untitled PDF"


def safe_filename(filename: str) -> str:
    cleaned = Path(filename or "document.pdf").name.replace(os.sep, "_")
    return cleaned if cleaned.lower().endswith(".pdf") else f"{cleaned}.pdf"


def public_document(document: dict) -> dict:
    return {
        "id": document["id"],
        "filename": document["filename"],
        "display_name": document["display_name"],
        "size": document["size"],
        "status": document["status"],
        "pages": document.get("pages", 0),
        "chunks": document.get("chunks", 0),
        "error": document.get("error"),
    }


def public_documents_for(saved_files: List[dict]) -> List[dict]:
    with state_lock:
        return [
            public_document(document)
            for item in saved_files
            if (document := documents.get(item["id"]))
        ]


def ensure_vectorstore() -> Chroma:
    global vectorstore

    with state_lock:
        if vectorstore is None:
            import chromadb

            client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
            vectorstore = Chroma(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
            )
        return vectorstore


def processed_documents_count() -> int:
    with state_lock:
        return sum(1 for document in documents.values() if document["status"] == "processed")


def update_job(job_id: str, **updates) -> None:
    with state_lock:
        job = upload_jobs.get(job_id)
        if job:
            job.update(updates)
            job["updated_at"] = time.time()


def update_document(document_id: str, **updates) -> None:
    with state_lock:
        document = documents.get(document_id)
        if document:
            document.update(updates)
            document["updated_at"] = time.time()


def remove_document_from_jobs(document_id: str, filename: str) -> None:
    with state_lock:
        for job in upload_jobs.values():
            job_documents = [
                item for item in job.get("documents", []) if item.get("id") != document_id
            ]
            if len(job_documents) != len(job.get("documents", [])):
                job["documents"] = job_documents
                job["total_files"] = len(job_documents)
                if job.get("active_file") == filename:
                    job["active_file"] = None
                job["updated_at"] = time.time()


def job_cancelled(job_id: str) -> bool:
    with state_lock:
        return upload_jobs.get(job_id, {}).get("status") == "cancelled"


def process_upload_job(job_id: str, saved_files: List[dict]) -> None:
    update_job(job_id, status="processing", started_at=time.time())
    store = ensure_vectorstore()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250,
        separators=["\n\n", "\n", " ", ""],
    )

    processed_files = 0
    failed_files = 0
    total_chunks = 0
    total_pages = 0

    for file_info in saved_files:
        if job_cancelled(job_id):
            return

        document_id = file_info["id"]
        file_path = file_info["path"]
        filename = file_info["filename"]
        name = file_info["display_name"]
        update_document(document_id, status="processing")
        update_job(job_id, active_file=filename)

        try:
            loader = PyPDFLoader(file_path)
            current_batch = []
            pages_for_file = 0
            chunks_for_file = 0

            for page in loader.lazy_load():
                if job_cancelled(job_id):
                    return

                pages_for_file += 1
                page.metadata.update(
                    {
                        "document_id": document_id,
                        "source": filename,
                        "display_name": name,
                    }
                )
                current_batch.append(page)

                if len(current_batch) >= BATCH_SIZE:
                    chunks = splitter.split_documents(current_batch)
                    with state_lock:
                        store.add_documents(chunks)
                    chunks_for_file += len(chunks)
                    total_chunks += len(chunks)
                    current_batch = []
                    update_document(
                        document_id,
                        pages=pages_for_file,
                        chunks=chunks_for_file,
                    )
                    update_job(
                        job_id,
                        processed_pages=total_pages + pages_for_file,
                        total_chunks=total_chunks,
                    )

            if current_batch:
                chunks = splitter.split_documents(current_batch)
                with state_lock:
                    store.add_documents(chunks)
                chunks_for_file += len(chunks)
                total_chunks += len(chunks)

            processed_files += 1
            total_pages += pages_for_file
            update_document(
                document_id,
                status="processed",
                pages=pages_for_file,
                chunks=chunks_for_file,
            )
            update_job(
                job_id,
                processed_files=processed_files,
                processed_pages=total_pages,
                total_chunks=total_chunks,
                documents=public_documents_for(saved_files),
            )

        except Exception as exc:
            failed_files += 1
            update_document(document_id, status="failed", error=str(exc))
            update_job(job_id, error=str(exc))

    final_status = "completed"
    if failed_files and not processed_files:
        final_status = "failed"
    elif failed_files:
        final_status = "completed_with_errors"

    update_job(
        job_id,
        status=final_status,
        completed_at=time.time(),
        processed_files=processed_files,
        failed_files=failed_files,
        processed_pages=total_pages,
        total_chunks=total_chunks,
        documents=public_documents_for(saved_files),
    )


def cosine_similarity(left: List[float], right: List[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def retrieve_and_rerank(question: str, document_ids: Optional[List[str]] = None) -> List[tuple]:
    store = ensure_vectorstore()
    selected_ids = set(document_ids or [])

    if selected_ids:
        candidates = []
        per_document_k = max(
            6, math.ceil(RETRIEVAL_CANDIDATES / max(1, len(selected_ids)))
        )
        seen_keys = set()

        for document_id in selected_ids:
            with state_lock:
                matches = store.similarity_search(
                    question,
                    k=per_document_k,
                    filter={"document_id": document_id},
                )

            for doc in matches:
                key = (
                    doc.metadata.get("document_id"),
                    doc.metadata.get("page"),
                    doc.page_content[:120],
                )
                if key not in seen_keys:
                    seen_keys.add(key)
                    candidates.append(doc)
    else:
        with state_lock:
            candidates = store.similarity_search(question, k=RETRIEVAL_CANDIDATES)

    if not candidates:
        return []

    query_embedding = embeddings.embed_query(question)
    doc_embeddings = embeddings.embed_documents([doc.page_content for doc in candidates])
    scored_docs = []

    for doc, doc_embedding in zip(candidates, doc_embeddings):
        score = cosine_similarity(query_embedding, doc_embedding)
        scored_docs.append((doc, score))

    scored_docs.sort(key=lambda item: item[1], reverse=True)

    # Keep the strongest chunks while nudging the final context toward multiple files.
    reranked = []
    source_counts: Dict[str, int] = {}
    for doc, score in scored_docs:
        source_id = doc.metadata.get("document_id", "unknown")
        adjusted_score = score - (source_counts.get(source_id, 0) * 0.025)
        reranked.append((doc, adjusted_score))
        source_counts[source_id] = source_counts.get(source_id, 0) + 1

    reranked.sort(key=lambda item: item[1], reverse=True)
    return reranked[:RERANK_TOP_K]


def format_context(scored_docs: List[tuple]) -> str:
    context_parts = []
    for index, (doc, score) in enumerate(scored_docs, start=1):
        name = doc.metadata.get("display_name") or doc.metadata.get("source") or "Document"
        page = doc.metadata.get("page")
        page_label = f", page {page + 1}" if isinstance(page, int) else ""
        context_parts.append(
            f"[Source {index}: {name}{page_label}, relevance {score:.2f}]\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(context_parts)


def normalize_answer_tone(answer: str) -> str:
    text = answer.strip()
    if not text:
        return text

    text = re.sub(
        r"^\s*(According to|Based on|From) (the )?(provided |given )?context[:,]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^\s*The (provided )?context (says|mentions|explains|shows)( that)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^\s*The (provided )?context does not contain information about ([^.]+)\.\s*",
        r"I couldn't find anything about \2 in your PDF. ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^\s*The (provided )?context does not mention ([^.]+)\.\s*",
        r"I couldn't find anything about \2 in your PDF. ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:It|The (?:provided )?context) only discusses ([^.]+)\.\s*",
        r"This PDF only talks about \1. ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:It|The (?:provided )?context) only covers ([^.]+)\.\s*",
        r"This PDF only covers \1. ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bTherefore, I cannot provide [^.]+ based on (?:the )?(?:given|provided) context\.\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bIf you have any questions about ([^.]+), I'll be happy to help\.?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bbased on (?:the )?(?:given|provided) context\b",
        "from your PDF",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sanitized_history_snapshot() -> List:
    with state_lock:
        snapshot = chat_history[-MAX_HISTORY_MESSAGES:].copy()

    cleaned_history = []
    for message in snapshot:
        if isinstance(message, AIMessage):
            cleaned_history.append(AIMessage(content=normalize_answer_tone(str(message.content))))
        else:
            cleaned_history.append(message)
    return cleaned_history


def build_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are DocBot, a friendly AI assistant that answers questions using only the "
                "uploaded PDFs in this chat. Multiple PDFs may be available at once. Use the most "
                "relevant information across them and keep your tone warm, natural, and easy to "
                "understand for non-technical users.\n\n"
                "Important style rules:\n"
                "- Do not say phrases like 'According to the provided context', 'The context says', "
                "or 'Based on the given context'.\n"
                "- Answer like a normal helpful person.\n"
                "- Keep explanations simple and clear unless the user asks for more detail.\n"
                "- If the PDFs do not contain the answer, say so in a friendly way like "
                "'I couldn't find that in your PDF' and briefly say what the PDF does cover when useful.\n"
                "- Do not invent missing facts.\n"
                "- When helpful, mention the document name naturally, without sounding formal.\n\n"
                "Examples:\n"
                "- Instead of 'According to the provided context, React Native...', say "
                "'React Native lets you build mobile apps using JavaScript and React.'\n"
                "- Instead of 'The provided context does not contain information about Flutter', say "
                "'I couldn't find anything about Flutter in your PDF. This PDF only talks about React Native.'\n\n"
                "Context:\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )


def generate_answer(message: str, document_ids: Optional[List[str]] = None) -> str:
    if processed_documents_count() == 0:
        raise HTTPException(status_code=400, detail="No processed document is ready yet")

    scored_docs = retrieve_and_rerank(message, document_ids)
    prompt = build_prompt()
    history_snapshot = sanitized_history_snapshot()

    chain = prompt | llm | StrOutputParser()
    raw_answer = chain.invoke(
        {
            "context": format_context(scored_docs),
            "question": message,
            "chat_history": history_snapshot,
        }
    )
    answer = normalize_answer_tone(raw_answer)

    with state_lock:
        chat_history.append(HumanMessage(content=message))
        chat_history.append(AIMessage(content=answer))
        if len(chat_history) > MAX_HISTORY_MESSAGES:
            del chat_history[:-MAX_HISTORY_MESSAGES]

    return answer


def stream_answer(message: str, document_ids: Optional[List[str]] = None):
    scored_docs = retrieve_and_rerank(message, document_ids)
    prompt = build_prompt()

    history_snapshot = sanitized_history_snapshot()

    chain = prompt | llm | StrOutputParser()
    answer_parts: List[str] = []
    initial_buffer = ""
    did_flush_initial_buffer = False

    try:
        for token in chain.stream(
            {
                "context": format_context(scored_docs),
                "question": message,
                "chat_history": history_snapshot,
            }
        ):
            if token:
                if not did_flush_initial_buffer:
                    initial_buffer += token
                    if len(initial_buffer) < 220 and "." not in initial_buffer and "\n" not in initial_buffer:
                        continue

                    cleaned_start = normalize_answer_tone(initial_buffer)
                    did_flush_initial_buffer = True
                    initial_buffer = ""
                    if cleaned_start:
                        answer_parts.append(cleaned_start)
                        yield cleaned_start
                    continue

                answer_parts.append(token)
                yield token

        if not did_flush_initial_buffer and initial_buffer:
            cleaned_start = normalize_answer_tone(initial_buffer)
            if cleaned_start:
                answer_parts.append(cleaned_start)
                yield cleaned_start
    except Exception as exc:
        error_message = "\n\nI hit an error while streaming the answer. Please try again."
        answer_parts.append(error_message)
        print(f"Streaming error: {exc}")
        yield error_message
    finally:
        answer = normalize_answer_tone("".join(answer_parts).strip())
        if answer:
            with state_lock:
                chat_history.append(HumanMessage(content=message))
                chat_history.append(AIMessage(content=answer))
                if len(chat_history) > MAX_HISTORY_MESSAGES:
                    del chat_history[:-MAX_HISTORY_MESSAGES]


@app.get("/")
async def root():
    return {
        "message": "Welcome to DocBot API",
        "documents": processed_documents_count(),
    }


@app.post("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    files: Optional[List[UploadFile]] = File(default=None),
    file: Optional[List[UploadFile]] = File(default=None),
):
    upload_files = list(files or []) + list(file or [])
    if not upload_files:
        raise HTTPException(status_code=400, detail="Upload at least one PDF file")

    saved_files = []
    for upload in upload_files:
        original_name = safe_filename(upload.filename or "document.pdf")
        if not original_name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        document_id = uuid.uuid4().hex
        stored_name = f"{document_id}-{original_name}"
        file_path = UPLOAD_DIR / stored_name

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)

        document = {
            "id": document_id,
            "filename": original_name,
            "display_name": display_name(original_name),
            "size": file_path.stat().st_size,
            "path": str(file_path),
            "status": "queued",
            "pages": 0,
            "chunks": 0,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        with state_lock:
            documents[document_id] = document

        saved_files.append(
            {
                "id": document_id,
                "filename": original_name,
                "display_name": document["display_name"],
                "path": str(file_path),
            }
        )

    job_id = uuid.uuid4().hex
    job = {
        "job_id": job_id,
        "status": "queued",
        "total_files": len(saved_files),
        "processed_files": 0,
        "failed_files": 0,
        "processed_pages": 0,
        "total_chunks": 0,
        "active_file": None,
        "documents": public_documents_for(saved_files),
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    with state_lock:
        upload_jobs[job_id] = job

    background_tasks.add_task(process_upload_job, job_id, saved_files)
    return job


@app.get("/upload/status/{job_id}")
async def upload_status(job_id: str):
    with state_lock:
        job = upload_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Upload job not found")
        refreshed = dict(job)
        refreshed["documents"] = [
            public_document(documents[document["id"]])
            for document in job["documents"]
            if document["id"] in documents
        ]
        return refreshed


@app.get("/documents")
async def list_documents():
    with state_lock:
        return {"documents": [public_document(document) for document in documents.values()]}


@app.get("/documents/{document_id}/file")
async def document_file(document_id: str):
    with state_lock:
        document = documents.get(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        path = document["path"]
        filename = document["filename"]

    if not Path(path).exists():
        raise HTTPException(status_code=404, detail="Document file not found")

    return FileResponse(path, media_type="application/pdf", filename=filename)


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    global vectorstore, chat_history

    with state_lock:
        document = documents.get(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        path = document["path"]
        filename = document["filename"]
        documents.pop(document_id, None)
        remaining_documents = [public_document(item) for item in documents.values()]
        if not documents:
            chat_history = []

    remove_document_from_jobs(document_id, filename)

    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass

    try:
        if remaining_documents:
            store = ensure_vectorstore()
            with state_lock:
                store.delete(where={"document_id": document_id})
        else:
            import chromadb

            client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
            try:
                client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            with state_lock:
                vectorstore = None
    except Exception as exc:
        print(f"Warning: could not fully remove vectors for {document_id}: {exc}")

    return {
        "message": f"Document {filename} removed",
        "documents": remaining_documents,
    }


@app.delete("/documents")
async def clear_documents():
    global vectorstore, chat_history

    with state_lock:
        for job in upload_jobs.values():
            if job["status"] in {"queued", "processing"}:
                job["status"] = "cancelled"
                job["updated_at"] = time.time()

        documents.clear()
        chat_history = []

        import chromadb

        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        vectorstore = None

    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    return {"message": "All documents cleared"}


@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        return {"answer": generate_answer(request.message.strip(), request.document_ids)}
    except HTTPException:
        raise
    except Exception as exc:
        import traceback

        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if processed_documents_count() == 0:
        raise HTTPException(status_code=400, detail="No processed document is ready yet")

    return StreamingResponse(
        stream_answer(request.message.strip(), request.document_ids),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

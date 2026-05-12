import os

# SQLite3 fix for ChromaDB on Render
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

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
UPLOAD_DIR = "uploads"
CHROMA_DB_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("Configuring HuggingFace API Embeddings...")
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Initializing Groq LLM...")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
)

# Global state
vectorstore = None
retriever = None
chat_history: List = []   # stores HumanMessage / AIMessage objects

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []

@app.get("/")
async def root():
    return {"message": "Welcome to DocBot API"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore, retriever, chat_history
    print(f"Received file: {file.filename}")

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        print("1. Loading PDF...")
        loader = PyPDFLoader(file_path)
        
        print("2. Initializing ChromaDB...")
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        try:
            client.delete_collection("pdf_knowledge")
        except Exception:
            pass

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=250,
            separators=["\n\n", "\n", " ", ""]
        )

        # Batch processing for large documents (1000+ pages)
        print("3. Processing documents in batches...")
        vectorstore = None
        current_batch = []
        batch_size = 50 # Process 50 pages at a time
        
        for i, page in enumerate(loader.lazy_load()):
            current_batch.append(page)
            if len(current_batch) >= batch_size:
                chunks = text_splitter.split_documents(current_batch)
                if vectorstore is None:
                    vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        client=client,
                        collection_name="pdf_knowledge"
                    )
                else:
                    vectorstore.add_documents(chunks)
                current_batch = []
                print(f"   Processed {i+1} pages...")

        # Process remaining pages
        if current_batch:
            chunks = text_splitter.split_documents(current_batch)
            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    client=client,
                    collection_name="pdf_knowledge"
                )
            else:
                vectorstore.add_documents(chunks)
        
        print("4. Finalizing retriever...")
        # Use Maximal Marginal Relevance (MMR) for more diverse retrieval across the whole document
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 10,
                "fetch_k": 50,
                "lambda_mult": 0.5
            }
        )

        # Reset chat history when a new document is uploaded
        chat_history = []

        print("✅ Processing complete!")
        return {"message": f"File {file.filename} processed successfully"}

    except Exception as e:
        import traceback
        print("❌ ERROR DURING UPLOAD:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest):
    global retriever, llm, chat_history

    if retriever is None or llm is None:
        raise HTTPException(status_code=400, detail="No document uploaded yet")

    try:
        # Build the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are DocBot, a helpful AI assistant that answers questions strictly based on "
             "the provided document context. If the answer is not in the context, say so clearly.\n\n"
             "Context:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # LCEL chain
        chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda _: chat_history,
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke(request.message)

        # Save to history
        chat_history.append(HumanMessage(content=request.message))
        chat_history.append(AIMessage(content=answer))

        # Keep history manageable (last 10 exchanges = 20 messages)
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]

        return {"answer": answer}

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

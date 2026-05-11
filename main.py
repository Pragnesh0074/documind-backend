import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="DocuMind Backend")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = "uploads"
CHROMA_DB_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global state
vectorstore = None
retriever = None
llm = None
chat_history: List = []   # stores HumanMessage / AIMessage objects

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []

@app.get("/")
async def root():
    return {"message": "Welcome to DocuMind API"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore, retriever, llm, chat_history
    print(f"Received file: {file.filename}")

    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is missing. Please check your .env file.")

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        print("1. Loading PDF...")
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        print(f"2. Splitting {len(documents)} pages into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        print("3. Creating Local Embeddings (HuggingFace)...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        print("4. Storing in ChromaDB...")
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

        # Delete old collection if it exists to avoid schema conflicts
        try:
            client.delete_collection("pdf_knowledge")
        except Exception:
            pass

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=client,
            collection_name="pdf_knowledge"
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        print("5. Initializing Groq LLM (Llama 3.1)...")
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
        )

        # Reset chat history when a new document is uploaded
        chat_history = []

        print("✅ Processing complete!")
        return {"message": f"File {file.filename} processed successfully", "chunks": len(chunks)}

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
             "You are DocuMind, a helpful AI assistant that answers questions strictly based on "
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

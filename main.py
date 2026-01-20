import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load Environment Variables
load_dotenv()

app = FastAPI(title="Day 7: Streaming AI Chat")

# --- CONFIGURATION ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = PineconeVectorStore(index_name="rag-app", embedding=embeddings)

# IMPORTANT: enable `streaming=True` here
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    streaming=True
)

# --- DATA MODELS ---
class ChatRequest(BaseModel):
    question: str

# --- CORE LOGIC (Generator) ---
async def generate_chat_stream(question: str):
    """
    This function doesn't return a string.
    It YIELDS chunks of text one by one.
    """
    # 1. Retrieve Context
    docs = vectorstore.similarity_search(question, k=3)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    # 2. Prepare Prompt
    prompt_template = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Answer based on the context below.
    
    Context:
    {context}
    
    Question: 
    {question}
    """)
    
    chain = prompt_template | llm | StrOutputParser()
    
    # 3. Stream the Answer
    # .stream() creates an iterator that gives us text as it's generated
    async for chunk in chain.astream({
        "context": context_text,
        "question": question
    }):
        # We yield the chunk immediately to the browser
        yield chunk

# --- API ENDPOINT ---
@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    try:
        # We return a StreamingResponse that consumes our generator
        return StreamingResponse(
            generate_chat_stream(request.question),
            media_type="text/plain"
        )
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
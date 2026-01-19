import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load Environment Variables
load_dotenv()

app = FastAPI(title="Day 3: RAG Chat API")

# --- CONFIGURATION ---
# Setup Google Embeddings (Must match Day 2 setup)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Connect to Pinecone Index
# We don't upload data here; we just "connect" to the existing index
vectorstore = PineconeVectorStore(
    index_name="rag-app",
    embedding=embeddings
)

# Initialize the LLM (Gemini Flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# --- DATA MODELS ---
class ChatRequest(BaseModel):
    question: str = Field(description="The user's question about the document")

class ChatResponse(BaseModel):
    answer: str = Field(description="The AI's answer based on the document")
    sources: List[str] = Field(description="The exact text chunks used to answer")

# --- CORE LOGIC (RAG) ---
def get_rag_response(question: str) -> ChatResponse:
    # 1. RETRIEVE: Search Pinecone for the 3 most similar chunks
    # This turns the question into numbers and finds the "nearest neighbors"
    print(f"Searching for: {question}")
    docs = vectorstore.similarity_search(question, k=3)
    
    # 2. CONTEXT: Combine the retrieved chunks into one big string
    # This acts as the "background knowledge" for the AI
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    # 3. AUGMENT: Create the prompt with the context
    prompt_template = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Answer the user's question strictly based on the context provided below.
    If the answer is not in the context, say "I don't know based on the provided document."

    Context:
    {context}

    Question: 
    {question}
    """)

    # 4. GENERATE: Send everything to Gemini
    chain = prompt_template | llm | StrOutputParser()
    
    answer = chain.invoke({
        "context": context_text,
        "question": question
    })

    return ChatResponse(
        answer=answer,
        sources=[doc.page_content for doc in docs] # Return sources for debugging
    )

# --- API ENDPOINT ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_pdf(request: ChatRequest):
    try:
        response = get_rag_response(request.question)
        return response
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
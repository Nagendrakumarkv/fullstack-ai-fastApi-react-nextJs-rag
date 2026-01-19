import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# 1. Load Environment Variables
load_dotenv()

app = FastAPI(title="Day 4: RAG Chat with Memory")

# --- CONFIGURATION ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

vectorstore = PineconeVectorStore(
    index_name="rag-app",
    embedding=embeddings
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# --- MEMORY STORAGE (In-Memory) ---
# In a real app, you would use Redis here. For now, a Python dictionary works.
# Format: { "session_id_1": [Message1, Message2], "session_id_2": ... }
store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieves the chat history for a specific session ID.
    If it doesn't exist, it creates a new empty history.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- DATA MODELS ---
class ChatRequest(BaseModel):
    question: str = Field(description="The user's question")
    session_id: str = Field(description="A unique ID for this conversation (e.g., 'user_123')")

class ChatResponse(BaseModel):
    answer: str
    session_id: str

# --- CORE LOGIC (RAG + MEMORY) ---
# 1. Define the Prompt with History Placeholder
# --- UPDATED PROMPT (Allows Chit-Chat) ---
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
    
    1. If the user asks a question about the document, answer it using the Context below.
    2. If the user is just chatting (e.g., "Hi", "My name is..."), respond naturally and politely without using the Context.
    3. If the answer is not in the Context and not general knowledge, say "I don't know."
    
    Context:
    {context}"""),
    
    MessagesPlaceholder(variable_name="chat_history"),
    
    ("human", "{question}")
])

# 2. Create the Basic Chain
chain = prompt_template | llm | StrOutputParser()

# 3. Wrap the Chain with Memory capabilities
# This wrapper handles reading/writing to the 'store' dictionary automatically
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

def get_rag_response(question: str, session_id: str) -> str:
    # A. Search Pinecone (Retrieval)
    docs = vectorstore.similarity_search(question, k=3)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # B. Generate Answer with Memory
    # We pass 'context' manually, but 'chat_history' is handled by the wrapper
    response = chain_with_history.invoke(
        {"question": question, "context": context_text},
        config={"configurable": {"session_id": session_id}}
    )

    return response

# --- API ENDPOINT ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_memory(request: ChatRequest):
    try:
        answer_text = get_rag_response(request.question, request.session_id)
        return ChatResponse(
            answer=answer_text,
            session_id=request.session_id
        )
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
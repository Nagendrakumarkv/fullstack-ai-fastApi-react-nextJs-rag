import os
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pathlib import Path # <--- ADD THIS IMPORT

# --- THE FIX STARTS HERE ---
# Get the absolute path of the folder containing this script
current_dir = Path(__file__).parent
env_path = current_dir / ".env"

# Load the .env file from that specific path
load_dotenv(dotenv_path=env_path)
# --- THE FIX ENDS HERE ---

# 2. Initialize FastMCP
# This creates a server named "RAG-Server"
mcp = FastMCP("RAG-Knowledge-Base")

# 3. Setup the Connection to Pinecone (Reuse your logic from Day 2/3)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = PineconeVectorStore(index_name="rag-app", embedding=embeddings)

# --- DEFINE TOOLS ---

@mcp.tool()
def query_knowledge_base(question: str) -> str:
    """
    Search the candidate's Resume/PDF for specific information.
    Use this to find email, skills, experience, or project details.
    """

    # Perform the search
    docs = vectorstore.similarity_search(question, k=3)
    
    # Format the results
    context = "\n\n".join([doc.page_content for doc in docs])
    return f"Found relevant context:\n{context}"

@mcp.tool()
def calculate_salary_tax(salary: int) -> str:
    """
    A simple utility tool to calculate estimated tax (20% flat rate).
    Useful if the resume contains salary expectations.
    """
    tax = salary * 0.20
    net = salary - tax
    return f"Gross: ${salary}, Tax: ${tax}, Net: ${net}"

# 4. Run the Server
if __name__ == "__main__":
    # This command makes the server listen on Standard Input/Output (Stdio)
    # This is how MCP clients (like Claude Desktop) talk to servers.
    mcp.run()
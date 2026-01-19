import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# 1. Load Environment Variables
load_dotenv()

# 2. Setup Google Embeddings (The "Translator" to Numbers)
# We use 'models/text-embedding-004' which is Free and High Quality
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def ingest_docs():
    # --- Step 1: Load the PDF ---
    # Make sure you have a PDF named "my_cv.pdf" in your folder!
    print("Loading PDF...")
    loader = PyPDFLoader("my_cv.pdf") 
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} pages.")

    # --- Step 2: Split Text (Chunking) ---
    # We can't feed whole books to AI. We split them into "chunks".
    # chunk_size=1000: Each piece is ~1000 characters
    # chunk_overlap=200: Overlap ensures sentences aren't cut weirdly at the edges
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"Created {len(documents)} chunks.")

    # --- Step 3: Embed & Store in Pinecone ---
    print("Adding to Pinecone (this make take a few seconds)...")
    
    # This single line does the magic:
    # 1. Takes text chunks
    # 2. Sends them to Google to get numbers (Vectors)
    # 3. Uploads vectors to your Pinecone index named "rag-app"
    PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name="rag-app"
    )
    print("✅ Success! Documents are now vectors in the cloud.")

if __name__ == "__main__":
    ingest_docs()
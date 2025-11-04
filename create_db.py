# create_db.py
import os
from pathlib import Path

# ---------- IMPORTS (exact for your versions) ----------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# ---------- CONFIG ----------
PDF_PATH = "data.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PERSIST_DIR = "./chroma_db"
OLLAMA_MODEL = "llama3.1:8b"


def setup_chroma_db():
    """
    Loads PDF, splits, embeds, and creates or loads the Chroma DB, ensuring persistence.
    """
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)

    # Check if DB already exists
    # When loading, we use the Chroma constructor, not from_documents
    if os.path.exists(PERSIST_DIR) and any(os.listdir(PERSIST_DIR)):
        print(f"Chroma DB already exists at {PERSIST_DIR}. Loading existing DB.")
        # Load the existing DB
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
        return vectorstore

    print("--- Creating new Chroma DB ---")

    # 1. LOAD PDF
    try:
        loader = PyPDFLoader(PDF_PATH)
        all_pages = loader.load()
        pages = all_pages[:]
        print(f"Loaded {len(pages)} pages from {PDF_PATH}")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None

    # 2. SPLIT INTO CHUNKS
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks")

    # 3. CHROMA VECTORSTORE
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )

    vectorstore.persist()
    print(f"Created  {PERSIST_DIR}")
    return vectorstore


if __name__ == "__main__":
    setup_chroma_db()
    print("\n  DB saved.")
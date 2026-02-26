import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Configuration
INDEX_NAME = 'ucsd-courses'

def run_etl():
    print("--- Starting ETL Pipeline ---")

    # 1. Initialize Pinecone Cliend (to manage the index)
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Check if the index exists, if not create it
    # For. simple update strategy: "Delete and Replace"
    # This ensures we don't have the duplicate courses if we run the script twice
    index = pc.Index(INDEX_NAME)

    print("--- Clearing old data ... ---")
    try:
        index.delete(delete_all=True)
        print("--- Old data deleted.")
    except Exception as e:
        print(f"Index might be empty or error: {e}")
    
    # 2. EXTRACT (Scrape)
    print("--- Extracting data from UCSD Catalog... ---")
    loader = WebBaseLoader("https://catalog.ucsd.edu/courses/CSE.html")
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")

    # 3. TRANSFORM (Split)
    print("--- Transforming (Splitting) data... ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True,
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    # 4. LOAD (Vectorize & Store)
    print("--- Loading into Pinecone... ---")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # This uploads the vectors to cloud
    PineconeVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    print("--- ETL Complete! Data is live in the cloud. ---")

if __name__ == "__main__":
    run_etl()
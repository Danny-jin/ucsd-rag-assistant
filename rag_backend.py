import os  # for interact os EV (environment variables)
import bs4  # filter for HTML, wrapper
from dotenv import load_dotenv  # EV for API KEY

from langchain.chat_models import init_chat_model  # Set up chat model
from langchain_openai import OpenAIEmbeddings  # Set up embedding model
from langchain_chroma import Chroma  # Set up vector storage

# This tool visit URL and app;y strainer to extract text
from langchain_community.document_loaders import WebBaseLoader

# Break the text down for LLM easy read
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use tool to retreive info from vectorized database
from langchain_core.tools import tool
from langchain.agents import create_agent

# Implement hybird search, this prevents the failure when search specific keywords
# like CSE 156, combine with best_match 25 with existing chroma search using
# an EmsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# The Ensemble retriever may return 10 results, best answer might be at #7,
# Use cross encorder to rearrange them by relevance before sending to LLM
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

# Load environment variables
load_dotenv()


def get_agent():
    """
    Initializes the Vector Store, Embeddings, and Agent.
    This function will be cached by Streamlit.
    """
    print("--- Initializing Agent & Vector Store ---")

    # Set up chat model
    # Temperature set to zero to reduce hallucinate, since this app is aim for
    # course selection, deterministic is required
    model = init_chat_model("gpt-5-nano", model_provider="openai", temperature=0)
    # Set up embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Set up vector store
    PERSIST_DIR = "./chroma_db"
    vector_store = Chroma(
        collection_name="ucsd_courses",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    # Indexing: Load, split, store
    # We check if the DB is empty inside the function now
    if len(vector_store.get()["ids"]) == 0:
        print("--- Database is empty. Indexing content... ---")
        # Step 1: Loading
        # bs4 specificly look class named 'layout-main'
        # bs4_strainer = bs4.SoupStrainer(class_=("layout-main"))
        # Webbaseloader visit url and apply strainer to extract info
        loader = WebBaseLoader(
            web_path=("https://catalog.ucsd.edu/courses/CSE.html",),
            # bs_kwargs={"parse_only": bs4_strainer},
        )
        # Ececute the fetch, return a list of Document objects
        docs = loader.load()

        # Step 2: Splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Target chunk of 1000 char
            chunk_overlap=200,  # keep 200 char overlap between chunk
            add_start_index=True,  # Track index in original document
        )

        # Takes the list of document objects and run them in splitter
        all_splits = text_splitter.split_documents(docs)

        # Step 3: Storing
        # Save the chunks into vectored database
        vector_store.add_documents(documents=all_splits)
        print("--- Content indexed and saved to disk! ---")
    else:
        print("--- Database found on disk. Skipping download & embedding. ---")

    # 1. Define the retriever
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 2. Keyboard retriever (BM 25)
    # Note: BM 25 need to be init with document, for now we rebuilt it from vector
    # store documents
    doc_data = vector_store.get()  # Get all data from Chroma store
    # reconstruct simple document object for BM 25
    from langchain_core.documents import Document

    docs_for_bm25 = [
        Document(page_content=txt, metadata=meta)
        for txt, meta in zip(doc_data["documents"], doc_data["metadatas"])
    ]
    keyword_retriever = BM25Retriever.from_documents(docs_for_bm25)
    keyword_retriever.k = 5

    # 3. Hybird emsemble
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever], weights=[0.5, 0.5]
    )

    # Define the reranker
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")

    # Wrap the ensemble retriever with the compressor
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,  # Or just vector_retriever if you skipped step 2
    )

    # Define tool
    # @tool decorator convert python script to LangChain Tool Object
    @tool(response_format="content_and_artifact")
    # automatic use functions docstrings and type hints to generate schema
    # LLM use the schema to under stand what tool does and how to use it
    def retrieve_context(query: str):
        """Retrieve context about UCSD CSE courses to help answer the question."""
        # Convert users query into a vector and compare all vector in database
        # k = 2 specify only retrieve top 2 most relevant chunks
        # retrieved_docs = vector_store.similarity_search(query, k=5)
        retrieved_docs = compression_retriever.invoke(query)

        # the loop iterate through found documents and stitch them into single string
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    # Create agent
    # The agent need a list of capabilities, so must be passed as list
    tools = [retrieve_context]
    system_prompt = (
        "You are a helpful assistant for a UCSD student. "
        "You have access to a tool to retrieve content from the UCSD CSE course catalog. "
        "If the user asks a follow-up question (e.g., 'What are its prerequisites?'), "
        "use the chat history to identify which course they are referring to, "
        "and include the specific course code in your input to the tool. "
        "Do not assume the tool knows the chat history."
    )
    # The function help create the agent
    agent = create_agent(model, tools, system_prompt=system_prompt)

    return agent


# Only run this if executing the file directly (for testing)
if __name__ == "__main__":
    agent = get_agent()

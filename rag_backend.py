import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool

load_dotenv()

def get_agent():
    print("--- Initializing Agent with Pinecone ---")

    # 1. Setup Models
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 2. Connect to Vector Store (Read-only mode)
    # Here we no longer check if its empty or scrape
    # We assume etl_pipeline.py already handle that
    vector_store = PineconeVectorStore(
        index_name="ucsd-courses",
        embedding=embeddings
    )

    # 3. Define Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Note: For Step 2, let's stick to pure Vector Search to ensure Pinecone works.
    # We can re-add BM25 (Hybrid Search) later, but BM25 is tricky with Pinecone 
    # because Pinecone doesn't store the raw text index locally for BM25.
    # For now, we will simplify to just Vector Search.

    @tool
    def retrieve_context(query: str):
        """Retrieve context about UCSD CSE courses."""
        docs = retriever.invoke(query)
        return "\n\n".join([d.page_content for d in docs])
    
    # 4. Create Agent
    tools = [retrieve_context]
    system_prompt = (
        "You are a helpful assistant for UCSD students. "
        "Use the retrieve_context tool to find course info."
    )

    agent = create_react_agent(llm, tools, prompt=system_prompt)

    return agent
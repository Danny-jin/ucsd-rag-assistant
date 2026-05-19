# UCSD CSE Course Assistant 🎓

An **agentic Retrieval-Augmented Generation (RAG)** system that helps UCSD students explore Computer Science & Engineering (CSE) courses. Ask natural-language questions about prerequisites, course content, and scheduling — the system retrieves relevant catalog data from a vector store and grounds the LLM's answers in real sources.

---

## 🧭 Architecture Evolution (v1 → v2)

This project was built in two iterations. The journey is intentional — it started as a quick Streamlit prototype and was re-architected into a production-style backend.

### v1 — Streamlit Prototype (`old_outdated/`)
- Streamlit chat UI
- **ChromaDB** local vector store
- **Hybrid retrieval**: vector similarity + **BM25** keyword search
- **FlashRank** cross-encoder reranking
- Live demo (still running): https://ucsd-rag-assistant-r9lpymztcygugwr5bdqeta.streamlit.app/

### v2 — Agentic FastAPI Backend (current)
- Re-architected into a **FastAPI REST backend** (`api.py`)
- **Migrated** vector storage from local ChromaDB → cloud-native **Pinecone** (`etl_pipeline.py`)
- Replaced the static retrieval chain with a **LangGraph ReAct agent** that calls retrieval as a tool (`rag_backend.py`)
- **Containerized with Docker**, deployed on **AWS EC2** behind an **Nginx** reverse proxy

> **Note on v2 retrieval:** v2 uses pure vector search (no BM25). Pinecone is a managed vector index and doesn't expose a local raw-text index, so the BM25 hybrid approach from v1 doesn't transfer directly. This was a deliberate simplification trade-off when moving to cloud-native storage.

---

## 🏗️ Current Architecture (v2)

```
User query
   │
   ▼
FastAPI /chat endpoint              (api.py)
   │   • validates request (Pydantic)
   │   • passes query + chat history to the agent
   ▼
LangGraph ReAct Agent               (rag_backend.py)
   │   • reasons about the query
   │   • decides when to call the retrieve_context tool
   ▼
Pinecone Vector Store               (index: "ucsd-courses", top-k = 5)
   │   • returns the 5 most semantically similar catalog chunks
   ▼
gpt-5-nano                          synthesizes a grounded answer + sources
```

### Data Ingestion Pipeline (`etl_pipeline.py`)
```
UCSD CSE course catalog (catalog.ucsd.edu/courses/CSE.html)
   → WebBaseLoader            (Extract / scrape)
   → RecursiveCharacterTextSplitter   (Transform: chunk_size=500, overlap=100)
   → OpenAI text-embedding-3-small    (Embed)
   → Pinecone upsert          (Load; delete-and-replace to avoid duplicates)
```

---

## 🛠️ Tech Stack

| Layer | v2 (current) | v1 (prototype) |
|-------|--------------|----------------|
| Interface | FastAPI REST API | Streamlit UI |
| Orchestration | LangChain + LangGraph (ReAct agent) | LangChain |
| LLM & Embeddings | OpenAI `gpt-5-nano`, `text-embedding-3-small` | same |
| Vector DB | Pinecone (serverless) | ChromaDB (local) |
| Retrieval | Pure vector search (top-k) | Hybrid (vector + BM25) + FlashRank rerank |
| Ingestion | BeautifulSoup / WebBaseLoader | same |
| Deployment | Docker → AWS EC2 + Nginx | Streamlit Community Cloud |

---

## 📋 Prerequisites
- Python 3.11+
- An OpenAI API key
- A Pinecone API key (for v2)

## ⚙️ Setup (v2)

```bash
# 1. Clone
git clone https://github.com/Danny-jin/ucsd-rag-assistant.git
cd ucsd-rag-assistant

# 2. (Optional) virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the root:
```env
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
```

## ▶️ Usage (v2)

**Step 1 — Build the index** (scrape + embed + upload to Pinecone):
```bash
python etl_pipeline.py
```

**Step 2 — Run the API server:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Step 3 — Query the `/chat` endpoint:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the prerequisites for CSE 151B?", "history": []}'
```

### Run with Docker
```bash
docker build -t ucsd-course-assistant .
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -e PINECONE_API_KEY=$PINECONE_API_KEY -p 8000:8000 ucsd-course-assistant
```

---

## 🗂️ Repository Structure
```
.
├── api.py              # FastAPI app: /chat endpoint, request/response models, agent lifespan
├── rag_backend.py      # LangGraph ReAct agent + Pinecone retriever + gpt-5-nano
├── etl_pipeline.py     # Scrape → chunk → embed → upsert to Pinecone
├── Dockerfile          # Containerized uvicorn server (port 8000)
├── requirements.txt
└── old_outdated/       # v1 prototype (Streamlit + ChromaDB + BM25 + FlashRank)
```

---

## 🚧 Roadmap
- [ ] Evaluation pipeline: measure retrieval recall@k against a hand-curated test set
- [ ] Re-introduce hybrid search in v2 (e.g., Pinecone sparse-dense vectors)
- [ ] Expand ingestion beyond CSE to additional UCSD departments
- [ ] Add response caching + cost/latency tracking

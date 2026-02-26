from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from rag_backend import get_agent
from pydantic import BaseModel
from typing import List, Optional

# Global variable to hold the agent
agent_runner = {}

# Lifespan manager
# This runs before server starts accepting requests
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading the agent...")
    # Init the agent
    agent_runner["agent"] = get_agent()
    yield
    print("Shutting down the server...")
    agent_runner.clear()

# Init the app
app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "active", "message": "Welcome to the UCSD Course API"}

# Represent single message in the history
class Message(BaseModel):
    role: str
    content:str

# Represent the data sent to your API
class ChatRequest(BaseModel):
    query:str
    history:Optional[List[Message]] = []

# Represent the data returned by your API
class ChatResponse(BaseModel):
    answer:str
    source:List[str]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # 1. Get the agent from global storage
    agent = agent_runner["agent"]
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not loaded")
    
    # 2. Prepare the input for the agent
    # Convert Pydanic Message back to simple dict for LangChain
    chat_history = []
    for msg in request.history:
        chat_history.append({"role": msg.role, "content": msg.content})

    # Add the current user query to the history
    chat_history.append({"role": "user", "content": request.query})

    # 3. Invoke the agent
    # The agent returns a dict with keys like 'messages'
    try:
        result = agent.invoke({"messages": chat_history})

        # 4. Extract the final answer
        # The last message in the 'messages' list is the agent's response
        ai_message = result["messages"][-1]
        # The content is a string, but it might contain markdown
        # We need to extract the text and the sources
        final_answer = ai_message.content
        # Extract sources from the tool calls
        sources = []
        if "tool_calls" in ai_message:
            for tool_call in ai_message.tool_calls:
                if tool_call["name"] == "retrieve_context":
                    # The tool returns a tuple of (content, artifacts)
                    # We want the content, which is the first element
                    sources.append(tool_call["content"][0])
    
        return ChatResponse(answer=final_answer, source=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

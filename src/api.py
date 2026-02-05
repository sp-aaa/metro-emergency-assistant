import logging
import asyncio
import os
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import ollama

from src.config import api_config, ollama_config

# 1. Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Pydantic Models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User's query")
    history: list = Field(default_factory=list, description="Chat history")
    stream: bool = Field(default=False, description="Enable streaming response")

class ChatResponse(BaseModel):
    response: str
    context: list = Field(default_factory=list, description="Updated context/history")

# 3. Lifespan Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Check if Ollama is reachable
    try:
        # Simple list call to check connection
        ollama.list()
        logger.info(f"Connected to Ollama at {ollama_config.base_url}")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        logger.warning("Please ensure Ollama is running (ollama serve)")
    yield
    # Shutdown

# 4. Initialize FastAPI
app = FastAPI(
    title="UrbanTransit-Assistant API",
    description="API for Metro Emergency Response Assistant backed by Qwen2.5 + LoRA",
    version="1.0.0",
    lifespan=lifespan
)

# 5. Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.allow_origins,
    allow_credentials=True,
    allow_methods=api_config.allow_methods,
    allow_headers=api_config.allow_headers,
)

# 6. Mount Frontend (optional)
static_dir = os.path.join(os.path.dirname(__file__), "web")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def read_root():
        return FileResponse(os.path.join(static_dir, "index.html"))

    @app.get("/{path:path}")
    async def read_spa(path: str, request: Request):
        if path.startswith("api/") or path == "api":
            return JSONResponse(status_code=404, content={"detail": "Not Found"})
        return FileResponse(os.path.join(static_dir, "index.html"))

# 7. Helper Functions
def build_prompt(message: str, history: list) -> list:
    """Construct message history for Ollama"""
    messages = [{"role": "system", "content": ollama_config.system_prompt}]
    
    # Add history (assuming history is list of {"role":..., "content":...})
    # If history is just strings, we might need to adapt.
    # Here we assume client sends list of dicts.
    for msg in history:
        messages.append(msg)
        
    messages.append({"role": "user", "content": message})
    return messages

# 7. Endpoints
@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok", "service": "UrbanTransit-Assistant"}

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Standard chat endpoint (non-streaming)
    """
    try:
        messages = build_prompt(request.message, request.history)
        
        response = ollama.chat(
            model=ollama_config.model_name,
            messages=messages,
            options={
                "temperature": ollama_config.temperature,
                "top_p": ollama_config.top_p,
                "num_ctx": ollama_config.num_ctx,
            }
        )
        
        return ChatResponse(
            response=response['message']['content'],
            context=messages + [{"role": "assistant", "content": response['message']['content']}]
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )

@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint
    """
    try:
        messages = build_prompt(request.message, request.history)
        
        async def generate() -> AsyncGenerator[str, None]:
            stream = ollama.chat(
                model=ollama_config.model_name,
                messages=messages,
                stream=True,
                options={
                    "temperature": ollama_config.temperature,
                    "top_p": ollama_config.top_p,
                }
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
                    # Let the event loop flush chunks promptly
                    await asyncio.sleep(0)
        
        return StreamingResponse(
            generate(),
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        
    except Exception as e:
        logger.error(f"Error in stream endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=api_config.host, port=api_config.port)

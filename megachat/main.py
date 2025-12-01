import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import router
from app.core.config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    print("=€ Starting MegaChat API...")
    print(f"=Ê Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    print(f"=¾ Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
    print(f"=Ä  PostgreSQL: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}")
    print(f"> LLM Model: {settings.LLM_MODEL}")
    print(f"= Embedding Model: {settings.EMBEDDING_MODEL}")
    print(f"=È Reranker Model: {settings.RERANKER_MODEL}")

    yield

    # Shutdown
    print("=K Shutting down MegaChat API...")


# Create FastAPI app
app = FastAPI(
    title="MegaChat - Persian Sales Chatbot",
    description="RAG-based Persian sales chatbot with LangChain, Qdrant, and OpenAI",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1", tags=["MegaChat"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "(G E¯'†* .H4 "E/Ì/!",
        "version": "1.0.0",
        "status": "running"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
    )

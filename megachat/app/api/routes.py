from fastapi import APIRouter, Depends, HTTPException, status
from redis.asyncio import Redis
from qdrant_client import QdrantClient
from datetime import datetime
import uuid

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ProductCreateRequest,
    ProductResponse,
    Product,
)
from app.core.dependencies import get_redis, get_qdrant
from app.chains.rag_chain import RAGPipeline
from app.services.retriever import MultiQueryRetriever

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    redis: Redis = Depends(get_redis),
    qdrant: QdrantClient = Depends(get_qdrant),
):
    """
    Main chat endpoint.

    Process user query through RAG pipeline and return response.
    """
    try:
        # Initialize RAG pipeline
        pipeline = RAGPipeline(redis, qdrant)

        # Generate session_id if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Run pipeline
        result = await pipeline.run(
            query=request.text,
            user_id=request.user_id,
            session_id=session_id
        )

        # Extract products from reranked docs
        retrieved_products = None
        if result.get("reranked_docs"):
            retrieved_products = [doc.product for doc in result["reranked_docs"]]

        # Build response
        response = ChatResponse(
            response=result["final_response"],
            intent=result.get("intent", "general"),
            from_cache=result.get("from_cache", False),
            retrieved_products=retrieved_products,
            confidence=result.get("slots", {}).get("confidence") if result.get("slots") else None,
            session_id=session_id,
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check(
    redis: Redis = Depends(get_redis),
    qdrant: QdrantClient = Depends(get_qdrant),
):
    """
    Health check endpoint.

    Check status of all services: Redis, Qdrant.
    """
    services = {}

    # Check Redis
    try:
        await redis.ping()
        services["redis"] = True
    except Exception:
        services["redis"] = False

    # Check Qdrant
    try:
        qdrant.get_collections()
        services["qdrant"] = True
    except Exception:
        services["qdrant"] = False

    # Overall status
    all_healthy = all(services.values())
    status_str = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        status=status_str,
        timestamp=datetime.now(),
        services=services
    )


@router.post("/products", response_model=ProductResponse)
async def add_product(
    request: ProductCreateRequest,
    qdrant: QdrantClient = Depends(get_qdrant),
):
    """
    Add a product to the vector database.
    """
    try:
        retriever = MultiQueryRetriever(qdrant)
        retriever.add_product(request.product)

        return ProductResponse(
            product=request.product,
            message="E-5HD (' EHABÌ* '6'AG 4/"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adding product: {str(e)}"
        )


@router.get("/products/{product_id}", response_model=Product)
async def get_product(
    product_id: str,
    redis: Redis = Depends(get_redis),
):
    """
    Get a product by ID from cache or database.
    """
    try:
        from app.services.cache import CacheService

        cache_service = CacheService(redis)
        product_data = await cache_service.get_product(product_id)

        if not product_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="E-5HD ~Ì/' F4/"
            )

        return Product(**product_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving product: {str(e)}"
        )

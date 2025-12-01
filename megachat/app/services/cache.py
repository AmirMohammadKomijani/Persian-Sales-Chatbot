import hashlib
import json
from typing import Optional, Any
from redis.asyncio import Redis
from app.core.config import get_settings

settings = get_settings()


class CacheService:
    """Redis cache service for responses, embeddings, products, and sessions"""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    @staticmethod
    def hash_query(text: str) -> str:
        """Generate consistent hash for query text"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def get_cached_response(self, query: str) -> Optional[str]:
        """Get cached response for a query"""
        query_hash = self.hash_query(query)
        key = f"response:{query_hash}"
        cached = await self.redis.get(key)
        return cached

    async def set_cached_response(
        self, query: str, response: str, ttl: Optional[int] = None
    ) -> None:
        """Cache a response for a query"""
        query_hash = self.hash_query(query)
        key = f"response:{query_hash}"
        ttl = ttl or settings.CACHE_RESPONSE_TTL
        await self.redis.setex(key, ttl, response)

    async def get_cached_embedding(self, query: str) -> Optional[list]:
        """Get cached embedding for a query"""
        query_hash = self.hash_query(query)
        key = f"embedding:{query_hash}"
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None

    async def set_cached_embedding(
        self, query: str, embedding: list, ttl: Optional[int] = None
    ) -> None:
        """Cache an embedding for a query"""
        query_hash = self.hash_query(query)
        key = f"embedding:{query_hash}"
        ttl = ttl or settings.CACHE_EMBEDDING_TTL
        await self.redis.setex(key, ttl, json.dumps(embedding))

    async def get_product(self, product_id: str) -> Optional[dict]:
        """Get cached product by ID"""
        key = f"product:{product_id}"
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None

    async def set_product(
        self, product_id: str, product_data: dict, ttl: Optional[int] = None
    ) -> None:
        """Cache product data"""
        key = f"product:{product_id}"
        ttl = ttl or settings.CACHE_PRODUCT_TTL
        await self.redis.setex(key, ttl, json.dumps(product_data))

    async def get_session(self, user_id: str) -> Optional[dict]:
        """Get user session (conversation history)"""
        key = f"session:{user_id}"
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None

    async def set_session(
        self, user_id: str, session_data: dict, ttl: Optional[int] = None
    ) -> None:
        """Save user session"""
        key = f"session:{user_id}"
        ttl = ttl or settings.CACHE_SESSION_TTL
        await self.redis.setex(key, ttl, json.dumps(session_data))

    async def append_to_session(
        self, user_id: str, message: dict, ttl: Optional[int] = None
    ) -> None:
        """Append a message to user session"""
        session = await self.get_session(user_id) or {"messages": []}
        session["messages"].append(message)
        await self.set_session(user_id, session, ttl)

    async def clear_session(self, user_id: str) -> None:
        """Clear user session"""
        key = f"session:{user_id}"
        await self.redis.delete(key)

    async def delete_cache(self, key: str) -> None:
        """Delete a cache entry"""
        await self.redis.delete(key)

    async def clear_all(self) -> None:
        """Clear all cache (use with caution)"""
        await self.redis.flushdb()

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from redis.asyncio import Redis
from qdrant_client import QdrantClient

from app.models.schemas import RAGState, IntentType
from app.services.preprocessor import PersianPreprocessor
from app.services.intent import IntentDetector
from app.services.retriever import MultiQueryRetriever
from app.services.reranker import RerankerService
from app.services.generator import ResponseGenerator
from app.services.cache import CacheService


class RAGPipeline:
    """
    LangGraph-based RAG pipeline for Persian sales chatbot.

    Pipeline:
    1. Check cache
    2. Preprocess query
    3. Detect intent & extract slots
    4. Retrieve documents (if needed)
    5. Rerank documents
    6. Generate response
    7. Cache response
    """

    def __init__(self, redis_client: Redis, qdrant_client: QdrantClient):
        # Initialize services
        self.cache_service = CacheService(redis_client)
        self.preprocessor = PersianPreprocessor()
        self.intent_detector = IntentDetector()
        self.retriever = MultiQueryRetriever(qdrant_client)
        self.reranker = RerankerService()
        self.generator = ResponseGenerator()

        # Build graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build LangGraph state graph"""
        workflow = StateGraph(dict)

        # Add nodes
        workflow.add_node("check_cache", self.check_cache)
        workflow.add_node("preprocess", self.preprocess)
        workflow.add_node("detect_intent", self.detect_intent)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("rerank", self.rerank)
        workflow.add_node("generate", self.generate)
        workflow.add_node("cache_response", self.cache_response)

        # Define edges
        workflow.set_entry_point("check_cache")

        # Conditional: if cached, skip to end
        workflow.add_conditional_edges(
            "check_cache",
            self.should_use_cache,
            {
                "use_cache": END,
                "continue": "preprocess"
            }
        )

        workflow.add_edge("preprocess", "detect_intent")

        # Conditional: if greeting, skip retrieval
        workflow.add_conditional_edges(
            "detect_intent",
            self.should_retrieve,
            {
                "retrieve": "retrieve",
                "skip_retrieval": "generate"
            }
        )

        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", "cache_response")
        workflow.add_edge("cache_response", END)

        return workflow.compile()

    async def check_cache(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check if response is cached"""
        query = state["query"]
        cached_response = await self.cache_service.get_cached_response(query)

        if cached_response:
            state["final_response"] = cached_response
            state["from_cache"] = True

        return state

    def should_use_cache(self, state: Dict[str, Any]) -> str:
        """Decide whether to use cached response"""
        if state.get("from_cache", False):
            return "use_cache"
        return "continue"

    async def preprocess(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess query"""
        query = state["query"]
        normalized = self.preprocessor.preprocess(query)
        state["normalized_query"] = normalized
        return state

    async def detect_intent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect intent and extract slots"""
        original_query = state["query"]
        normalized_query = state["normalized_query"]

        parsed = self.intent_detector.parse_query(normalized_query, original_query)

        state["intent"] = parsed.intent
        state["slots"] = parsed.slots

        return state

    def should_retrieve(self, state: Dict[str, Any]) -> str:
        """Decide whether to retrieve documents"""
        intent = state.get("intent")

        # Skip retrieval for greetings
        if intent == IntentType.GREETING:
            return "skip_retrieval"

        return "retrieve"

    async def retrieve(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant documents"""
        query = state["normalized_query"]
        intent = state.get("intent")
        slots = state.get("slots")

        retrieved_docs = self.retriever.retrieve(query, intent, slots)
        state["retrieved_docs"] = retrieved_docs

        return state

    async def rerank(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Rerank retrieved documents"""
        query = state["normalized_query"]
        retrieved_docs = state.get("retrieved_docs", [])

        if retrieved_docs:
            reranked_docs = self.reranker.rerank(query, retrieved_docs)
            state["reranked_docs"] = reranked_docs
        else:
            state["reranked_docs"] = []

        return state

    async def generate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response"""
        query = state["query"]
        intent = state.get("intent", IntentType.GENERAL)
        reranked_docs = state.get("reranked_docs", [])

        response = self.generator.generate(query, intent, reranked_docs)
        state["final_response"] = response

        return state

    async def cache_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Cache the generated response"""
        query = state["query"]
        response = state["final_response"]

        await self.cache_service.set_cached_response(query, response)

        return state

    async def run(self, query: str, user_id: str, session_id: str = None) -> Dict[str, Any]:
        """
        Run the full RAG pipeline.

        Args:
            query: User query
            user_id: User ID
            session_id: Session ID (optional)

        Returns:
            State dict with final_response and metadata
        """
        # Initialize state
        initial_state = {
            "query": query,
            "user_id": user_id,
            "session_id": session_id,
            "from_cache": False,
        }

        # Run graph
        final_state = await self.graph.ainvoke(initial_state)

        return final_state

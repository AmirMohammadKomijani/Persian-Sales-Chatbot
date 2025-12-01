from typing import List
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from app.models.schemas import Product, RetrievedDocument, IntentType, Slots
from app.core.config import get_settings

settings = get_settings()


class MultiQueryRetriever:
    """
    Retriever with multi-query expansion and Qdrant vector search.
    Uses multilingual-e5-large for embeddings.
    """

    def __init__(self, qdrant_client: QdrantClient):
        self.qdrant = qdrant_client
        self.collection_name = settings.QDRANT_COLLECTION_NAME

        # Load embedding model
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

        # LLM for query expansion
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0.3,
            api_key=settings.OPENAI_API_KEY,
        )

        # Initialize collection if needed
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Ensure Qdrant collection exists"""
        try:
            collections = self.qdrant.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                # Get embedding dimension
                sample_embedding = self.embedding_model.encode("test")
                vector_size = len(sample_embedding)

                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    ),
                )
        except Exception as e:
            print(f"Error ensuring collection exists: {e}")

    def generate_query_variations(self, query: str, num_variations: int = None) -> List[str]:
        """
        Generate query variations using LLM.
        Returns original query + variations.
        """
        num_variations = num_variations or settings.QUERY_VARIATIONS

        prompt_template = PromptTemplate(
            input_variables=["query", "num"],
            template="""4E' Ì© /3*Ì'1 GH4EF/ G3*Ì/. Ì© 3H'D A'13Ì /1Ì'A* EÌ©FÌ/ H ('Ì/ {num} F3.G E*A'H* '2 "F 3H'D 1' (3'2Ì/ ©G GE'F E9FÌ 1' /'4*G ('4F/ 'E' (' ©DE'* E*A'H* (Ì'F 4HF/.

3H'D '5DÌ: {query}

D7A'K {num} F3.G E*A'H* '2 'ÌF 3H'D 1' (G 5H1* Ì© DÌ3* 4E'1G/'1 (FHÌ3Ì/:
"""
        )

        try:
            prompt = prompt_template.format(query=query, num=num_variations - 1)
            response = self.llm.invoke(prompt)

            # Parse variations from response
            variations = [query]  # Always include original
            lines = response.content.strip().split("\n")

            for line in lines:
                line = line.strip()
                # Remove numbering (1., 2., -, etc.)
                if line and (line[0].isdigit() or line[0] in ["-", """, "*"]):
                    # Extract text after numbering
                    parts = line.split(".", 1) if "." in line else line.split(")", 1) if ")" in line else [line]
                    if len(parts) > 1:
                        variation = parts[1].strip()
                    else:
                        variation = parts[0].strip().lstrip("-"* ")

                    if variation and variation != query:
                        variations.append(variation)

            return variations[:num_variations]

        except Exception as e:
            print(f"Error generating query variations: {e}")
            return [query]

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using multilingual-e5-large"""
        # Add prefix for e5 model
        prefixed_text = f"query: {text}"
        embedding = self.embedding_model.encode(prefixed_text)
        return embedding.tolist()

    def search(
        self,
        query_variations: List[str],
        top_k: int = None,
        filters: dict = None
    ) -> List[RetrievedDocument]:
        """
        Search Qdrant with multiple query variations.
        Combines results from all variations.
        """
        top_k = top_k or settings.RETRIEVAL_TOP_K
        all_results = {}

        for query in query_variations:
            embedding = self.embed_text(query)

            # Build filter if provided
            query_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                if conditions:
                    query_filter = Filter(must=conditions)

            # Search
            try:
                results = self.qdrant.search(
                    collection_name=self.collection_name,
                    query_vector=embedding,
                    limit=top_k,
                    query_filter=query_filter,
                )

                # Aggregate results (deduplicate and keep best scores)
                for result in results:
                    product_id = result.id
                    if product_id not in all_results or result.score > all_results[product_id].score:
                        # Parse payload to Product
                        product = Product(**result.payload)
                        all_results[product_id] = RetrievedDocument(
                            product=product,
                            score=result.score
                        )

            except Exception as e:
                print(f"Error searching with query '{query}': {e}")
                continue

        # Sort by score and return top_k
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.score,
            reverse=True
        )[:top_k]

        # Add rank
        for i, doc in enumerate(sorted_results):
            doc.rank = i + 1

        return sorted_results

    def retrieve(
        self,
        query: str,
        intent: IntentType = None,
        slots: Slots = None,
    ) -> List[RetrievedDocument]:
        """
        Full retrieval pipeline:
        1. Generate query variations
        2. Search with all variations
        3. Return top_k results
        """
        # Generate variations
        variations = self.generate_query_variations(query)

        # Build filters from slots
        filters = {}
        if slots:
            if slots.brand:
                filters["brand"] = slots.brand
            if slots.color:
                filters["color"] = slots.color

        # Search
        results = self.search(variations, filters=filters if filters else None)

        return results

    def add_product(self, product: Product) -> None:
        """Add a product to Qdrant collection"""
        # Generate embedding from product text
        product_text = f"{product.name} {product.description or ''} {product.brand or ''}"
        embedding = self.embed_text(product_text)

        # Create point
        point = PointStruct(
            id=product.id,
            vector=embedding,
            payload=product.model_dump(mode="json")
        )

        # Upsert to collection
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

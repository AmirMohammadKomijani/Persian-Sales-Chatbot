from typing import List
from sentence_transformers import CrossEncoder
from app.models.schemas import RetrievedDocument
from app.core.config import get_settings

settings = get_settings()


class RerankerService:
    """
    Rerank retrieved documents using cross-encoder.
    Uses cross-encoder/ms-marco-multilingual-MiniLM-L12-v2.
    """

    def __init__(self):
        # Load cross-encoder model
        self.model = CrossEncoder(settings.RERANKER_MODEL)

    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int = None
    ) -> List[RetrievedDocument]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: User query
            documents: Retrieved documents to rerank
            top_k: Number of top documents to return

        Returns:
            List of reranked documents
        """
        if not documents:
            return []

        top_k = top_k or settings.RERANK_TOP_K

        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            # Create document text from product
            doc_text = f"{doc.product.name}"
            if doc.product.description:
                doc_text += f" {doc.product.description}"
            if doc.product.brand:
                doc_text += f" {doc.product.brand}"

            pairs.append([query, doc_text])

        # Score pairs with cross-encoder
        scores = self.model.predict(pairs)

        # Update scores in documents
        for i, doc in enumerate(documents):
            doc.score = float(scores[i])

        # Sort by new scores
        reranked = sorted(documents, key=lambda x: x.score, reverse=True)

        # Take top_k
        reranked = reranked[:top_k]

        # Update ranks
        for i, doc in enumerate(reranked):
            doc.rank = i + 1

        return reranked

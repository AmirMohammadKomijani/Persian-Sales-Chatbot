from hazm import Normalizer, word_tokenize
from typing import List


class PersianPreprocessor:
    """Persian text preprocessor using Hazm"""

    def __init__(self):
        self.normalizer = Normalizer()

    def normalize(self, text: str) -> str:
        """
        Normalize Persian text:
        - Fix Arabic/Persian characters (Ì/J ©/C)
        - Remove extra whitespace
        - Normalize punctuation
        """
        if not text:
            return ""

        # Apply Hazm normalization
        normalized = self.normalizer.normalize(text)

        # Remove extra whitespace
        normalized = " ".join(normalized.split())

        return normalized

    def tokenize(self, text: str) -> List[str]:
        """Tokenize Persian text"""
        if not text:
            return []

        normalized = self.normalize(text)
        tokens = word_tokenize(normalized)
        return tokens

    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline:
        1. Normalize text
        2. Clean up
        """
        normalized = self.normalize(text)
        return normalized

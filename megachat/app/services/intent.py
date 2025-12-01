import re
from typing import Dict, List
from app.models.schemas import IntentType, Slots, ParsedQuery


class IntentDetector:
    """Detect user intent and extract slots from Persian queries"""

    def __init__(self):
        # Intent patterns (Persian)
        self.intent_patterns = {
            IntentType.PRICE_CHECK: [
                r"BÌE*",
                r"†B/1",
                r"†F/ *HE'F",
                r"†F/G",
                r"G2ÌFG",
                r"†F/ ~HD",
            ],
            IntentType.AVAILABILITY: [
                r"EH,H/",
                r"/1 /3*13",
                r"/'1Ì/",
                r"G3*",
                r"H,H/ /'1",
                r"EÌ\s**HFE\s+(.1E",
            ],
            IntentType.FEATURE_INQUIRY: [
                r"E4.5'*",
                r"HÌ˜¯Ì",
                r"/H1(ÌF",
                r"('*1Ì",
                r"-'A8G",
                r"1E",
                r"E¯'~Ì©3D",
                r"'ÌF†",
                r"F3.G",
                r"E/D",
            ],
            IntentType.COMPARISON: [
                r"A1B",
                r"*A'H*",
                r"EB'Ì3G",
                r"(G*1",
                r"Ì'",
                r"©/HE",
                r"©/'E",
            ],
            IntentType.SHIPPING: [
                r"'13'D",
                r"*-HÌD",
                r"†F/ 1H2",
                r"2E'F",
                r"~3*",
                r"EÌ\s*13G",
                r"EÌ13G",
            ],
            IntentType.PURCHASE: [
                r"EÌ\s*.H'E\s+(.1E",
                r".1Ì/",
                r"3A'14",
                r"(.1E",
            ],
            IntentType.GREETING: [
                r"3D'E",
                r"/1H/",
                r"5(- (.Ì1",
                r"951 (.Ì1",
                r"EEFHF",
                r"*4©1",
            ],
        }

        # Slot extraction patterns
        self.slot_patterns = {
            "quantity": r"(\d+)\s*(9//|*')",
            "price": r"(\d+)\s*(*HE'F|EÌDÌHF|G2'1)",
            "color": r"(3AÌ/|3Ì'G|E4©Ì|B1E2|"(Ì|3(2|7D'ÌÌ|FB1G\s*'Ì|5H1*Ì)",
            "brand": r"(3'E3HF¯|'~D|4Ì'&HEÌ|'D\s*,Ì|'Ì3H3|FH©Ì'|GH'HÌ)",
        }

    def detect_intent(self, text: str) -> IntentType:
        """
        Detect user intent from text.
        Returns the intent with highest match score.
        """
        text_lower = text.lower()
        scores = {}

        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            scores[intent] = score

        # Return intent with highest score
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)

        # Default to GENERAL if no match
        return IntentType.GENERAL

    def extract_slots(self, text: str) -> Slots:
        """Extract slots (entities) from text"""
        slots = Slots()

        # Extract quantity
        quantity_match = re.search(self.slot_patterns["quantity"], text)
        if quantity_match:
            slots.quantity = int(quantity_match.group(1))

        # Extract color
        color_match = re.search(self.slot_patterns["color"], text)
        if color_match:
            slots.color = color_match.group(1)

        # Extract brand
        brand_match = re.search(self.slot_patterns["brand"], text)
        if brand_match:
            slots.brand = brand_match.group(1)

        # Extract price range
        price_matches = re.findall(self.slot_patterns["price"], text)
        if price_matches:
            prices = []
            for match in price_matches:
                price = int(match[0])
                unit = match[1]
                if "EÌDÌHF" in unit:
                    price *= 1_000_000
                elif "G2'1" in unit:
                    price *= 1_000
                prices.append(price)

            if len(prices) == 1:
                slots.price_range = {"min": 0, "max": prices[0]}
            elif len(prices) >= 2:
                slots.price_range = {"min": min(prices), "max": max(prices)}

        # Extract comparison items (simple approach)
        if "Ì'" in text or "H" in text:
            # This is a simplified extraction
            # In production, you'd use NER for better extraction
            tokens = text.split()
            comparison_items = []
            for i, token in enumerate(tokens):
                if token in ["Ì'", "H"] and i > 0 and i < len(tokens) - 1:
                    comparison_items.extend([tokens[i - 1], tokens[i + 1]])
            if comparison_items:
                slots.comparison_items = list(set(comparison_items))

        return slots

    def calculate_confidence(self, intent: IntentType, slots: Slots) -> float:
        """Calculate confidence score based on intent detection and slot extraction"""
        confidence = 0.5  # Base confidence

        # Increase confidence if intent is not GENERAL
        if intent != IntentType.GENERAL:
            confidence += 0.2

        # Increase confidence for each extracted slot
        slot_count = sum([
            1 if slots.product_name else 0,
            1 if slots.quantity else 0,
            1 if slots.color else 0,
            1 if slots.brand else 0,
            1 if slots.price_range else 0,
            1 if slots.comparison_items else 0,
        ])
        confidence += min(slot_count * 0.1, 0.3)

        return min(confidence, 1.0)

    def parse_query(self, normalized_text: str, original_text: str) -> ParsedQuery:
        """
        Parse user query and extract intent and slots.
        """
        intent = self.detect_intent(normalized_text)
        slots = self.extract_slots(normalized_text)
        confidence = self.calculate_confidence(intent, slots)

        return ParsedQuery(
            original_text=original_text,
            normalized_text=normalized_text,
            intent=intent,
            slots=slots,
            confidence=confidence,
        )

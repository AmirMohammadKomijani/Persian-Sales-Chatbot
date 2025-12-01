from typing import List
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from app.models.schemas import IntentType, RetrievedDocument
from app.core.config import get_settings

settings = get_settings()


class ResponseGenerator:
    """
    Generate Persian responses using LLM with retrieved context.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            api_key=settings.OPENAI_API_KEY,
        )

        # Intent-specific prompt templates
        self.templates = {
            IntentType.PRICE_CHECK: """4E' Ì© A1H4F/G -1AG'Ì G3*Ì/. ©'1(1 /1 EH1/ BÌE* E-5HD 3H'D ©1/G '3*.

3H'D ©'1(1: {query}

E-5HD'* E1*(7:
{context}

D7A'K Ì© ~'3. EAÌ/ H /BÌB (G A'13Ì (/GÌ/ ©G 4'ED BÌE* E-5HD'* ('4/. '¯1 †F/ E-5HD E1*(7 H,H/ /'1/ GEG 1' 0©1 ©FÌ/.
~'3. .H/ 1' ©H*'G H'6- H /H3*'FG (FHÌ3Ì/.""",

            IntentType.AVAILABILITY: """4E' Ì© A1H4F/G -1AG'Ì G3*Ì/. ©'1(1 /1 EH1/ EH,H/Ì E-5HD 3H'D ©1/G '3*.

3H'D ©'1(1: {query}

E-5HD'* E1*(7:
{context}

D7A'K Ì© ~'3. EAÌ/ (G A'13Ì (/GÌ/ ©G H69Ì* EH,H/Ì 1' E4.5 ©F/.
'¯1 E-5HD EH,H/ '3* 'ÌF 1' (G ©'1(1 '7D'9 /GÌ/. '¯1 EH,H/ FÌ3* ~Ì4FG'/G'Ì ,'Ì¯2ÌF '1'&G ©FÌ/.
~'3. .H/ 1' ©H*'G H H'6- (FHÌ3Ì/.""",

            IntentType.FEATURE_INQUIRY: """4E' Ì© A1H4F/G -1AG'Ì G3*Ì/. ©'1(1 /1 EH1/ E4.5'* H HÌ˜¯ÌG'Ì E-5HD 3H'D ©1/G '3*.

3H'D ©'1(1: {query}

E-5HD'* E1*(7:
{context}

D7A'K Ì© ~'3. ,'E9 (G A'13Ì (/GÌ/ ©G HÌ˜¯ÌG'Ì EGE E-5HD 1' *H6Ì- /G/.
(1 1HÌ HÌ˜¯ÌG'ÌÌ ©G ©'1(1 3H'D ©1/G *E1©2 ©FÌ/.
~'3. .H/ 1' H'6- H EAÌ/ (FHÌ3Ì/.""",

            IntentType.COMPARISON: """4E' Ì© A1H4F/G -1AG'Ì G3*Ì/. ©'1(1 EÌ.H'G/ †F/ E-5HD 1' (' GE EB'Ì3G ©F/.

3H'D ©'1(1: {query}

E-5HD'* E1*(7:
{context}

D7A'K Ì© EB'Ì3G /BÌB H (Ì71A'FG (G A'13Ì '1'&G /GÌ/.
*A'H*G'Ì ©DÌ/Ì 1' E4.5 ©FÌ/ H (G ©'1(1 ©E© ©FÌ/ *' (G*1ÌF 'F*.'( 1' /'4*G ('4/.
~'3. .H/ 1' 3'.*'1Ì'A*G H B'(D AGE (FHÌ3Ì/.""",

            IntentType.SHIPPING: """4E' Ì© A1H4F/G -1AG'Ì G3*Ì/. ©'1(1 /1 EH1/ '13'D H *-HÌD 3H'D ©1/G '3*.

3H'D ©'1(1: {query}

E-5HD'* E1*(7:
{context}

D7A'K '7D'9'* /BÌB /1 EH1/ 2E'F H F-HG '13'D 1' (G A'13Ì '1'&G /GÌ/.
'¯1 '7D'9'* '13'D /1 /'/GG' EH,H/ FÌ3* 41'Ì7 9EHEÌ '13'D 1' *H6Ì- /GÌ/ (E9EHD'K 2-3 1H2 ©'1Ì).
~'3. .H/ 1' H'6- (FHÌ3Ì/.""",

            IntentType.PURCHASE: """4E' Ì© A1H4F/G -1AG'Ì G3*Ì/. ©'1(1 EÌ.H'G/ E-5HD 1' .1Ì/'1Ì ©F/.

3H'D ©'1(1: {query}

E-5HD'* E1*(7:
{context}

D7A'K Ì© ~'3. EAÌ/ (G A'13Ì (/GÌ/ ©G '7D'9'* E-5HD BÌE* H F-HG .1Ì/ 1' 4'ED 4H/.
©'1(1 1' (1'Ì *©EÌD .1Ì/ 1'GFE'ÌÌ ©FÌ/.
~'3. .H/ 1' /H3*'FG H *4HÌB©FF/G (FHÌ3Ì/.""",

            IntentType.GREETING: """4E' Ì© A1H4F/G -1AG'Ì G3*Ì/. ©'1(1 (' 4E' '-H'D~13Ì ©1/G '3*.

~Ì'E ©'1(1: {query}

D7A'K Ì© ~'3. ¯1E H /H3*'FG (G A'13Ì (/GÌ/.
.H/ 1' E91AÌ ©FÌ/ H (~13Ì/ †¯HFG EÌ*H'FÌ/ ©E© ©FÌ/.
~'3. .H/ 1' ©H*'G H 5EÌEÌ (FHÌ3Ì/.""",

            IntentType.GENERAL: """4E' Ì© A1H4F/G -1AG'Ì G3*Ì/. ©'1(1 Ì© 3H'D 9EHEÌ ~13Ì/G '3*.

3H'D ©'1(1: {query}

E-5HD'* E1*(7:
{context}

D7A'K (G*1ÌF ~'3. EE©F 1' (G A'13Ì (/GÌ/.
'¯1 E-5HD'* E1*(7Ì ~Ì/' 4/ "FG' 1' E91AÌ ©FÌ/.
'¯1 FÌ'2 (G '7D'9'* (Ì4*1 /'1Ì/ '2 ©'1(1 (~13Ì/.
~'3. .H/ 1' EAÌ/ H /H3*'FG (FHÌ3Ì/.""",
        }

    def format_context(self, documents: List[RetrievedDocument]) -> str:
        """Format retrieved documents as context"""
        if not documents:
            return "GÌ† E-5HD E1*(7Ì ~Ì/' F4/."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            product = doc.product
            part = f"{i}. {product.name}"

            if product.price:
                part += f" - BÌE*: {product.price:,.0f} {product.currency}"

            if product.brand:
                part += f" - (1F/: {product.brand}"

            if product.availability:
                part += " - EH,H/"
            else:
                part += " - F'EH,H/"

            if product.description:
                part += f"\n   *H6Ì-'*: {product.description}"

            if product.features:
                features_str = ", ".join([f"{k}: {v}" for k, v in product.features.items()])
                part += f"\n   HÌ˜¯ÌG': {features_str}"

            context_parts.append(part)

        return "\n\n".join(context_parts)

    def generate(
        self,
        query: str,
        intent: IntentType,
        documents: List[RetrievedDocument]
    ) -> str:
        """
        Generate response using LLM.

        Args:
            query: User query
            intent: Detected intent
            documents: Retrieved and reranked documents

        Returns:
            Generated response in Persian
        """
        # Get appropriate template
        template_str = self.templates.get(intent, self.templates[IntentType.GENERAL])

        # Format context
        context = self.format_context(documents)

        # Create prompt
        prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=template_str
        )

        formatted_prompt = prompt.format(query=query, context=context)

        # Generate response
        try:
            response = self.llm.invoke(formatted_prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return "E*#3A'FG /1 -'D -'61 FEÌ*H'FE (G 3H'D 4E' ~'3. /GE. D7A'K /H('1G *D'4 ©FÌ/."

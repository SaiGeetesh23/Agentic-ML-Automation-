import re
from langchain_openai import ChatOpenAI


class LLMAgent:
    """
    Natural Language Interpreter
    - Extracts feature=value pairs
    - Converts messy human text into clean dict
    - Works for ALL datasets
    """

    def __init__(self, api_key):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=api_key
        )
 
    def extract_pairs_regex(self, text):
        pairs = re.findall(r'([A-Za-z0-9_()]+)\s*=\s*([-0-9.]+)', text)
        return {k: float(v) for k, v in pairs}

    def extract_with_llm(self, text):
        prompt = f"""
Extract all measurable feature values from the following user text.

User text:
{text}

Return ONLY a JSON object with key:value pairs.
Do NOT explain anything.
"""
        try:
            resp = self.llm.invoke(prompt).content
            import json
            return json.loads(resp)
        except:
            return {}

    
    def parse_features(self, text):
        """
        - First try regex
        - If no matches, use LLM extraction
        """
        pairs = self.extract_pairs_regex(text)
        if pairs:
            return pairs
        return self.extract_with_llm(text)

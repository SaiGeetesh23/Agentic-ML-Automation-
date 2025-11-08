# # agents/llm_agent.py
# import os
# import json
# from langchain_openai import ChatOpenAI
# # from langchain.chains import LLMChain
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from dotenv import load_dotenv
# import os

# load_dotenv()


# class LLMQueryAgent:
#     """
#     Uses an LLM to convert a natural language prediction request into a JSON mapping
#     of feature_name -> value. The prompt enforces output to be JSON only.
#     """

#     PROMPT = PromptTemplate(
#         input_variables=["features", "user_query"],
#         template="""You are a JSON extractor. Given the dataset features and a user query asking for a prediction,
# extract the requested feature values and return a JSON object with those features (only). 
# - If the user omits some features, return null for that feature.
# - Values must be numbers if numeric, or strings if obviously categorical.
# Return strictly a JSON object with keys exactly equal to the feature names provided.

# Features: {features}

# User query: {user_query}

# Output (JSON only):"""
#     )

#     def __init__(self, model_name="gpt-4o-mini", temperature=0.0):
#         api_key = os.environ.get("OPENAI_API_KEY")
#         if api_key is None:
#             raise RuntimeError("Set OPENAI_API_KEY in environment to use LLMQueryAgent")
#         self.llm = ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=os.environ.get("OPENAI_API_KEY"))
#         self.runnable_chain = self.PROMPT | self.llm | StrOutputParser()

#     def parse_query(self, features, user_query):
#         """
#         features: list of feature names (strings)
#         user_query: raw NL input
#         returns dict mapping feature->value (value may be null)
#         """
#         features_str = ", ".join(features)
#         resp = self.runnable_chain.invoke({"features": features_str, "user_query": user_query})

#         # Ensure JSON parse
#         try:
#             parsed = json.loads(resp)
#             # align keys
#             result = {}
#             for f in features:
#                 result[f] = parsed.get(f, None)
#             return result
#         except Exception:
#             # fallback: try to sanitize naive key:value pairs
#             return self._fallback_parse(resp, features)

#     def _fallback_parse(self, text, features):
#         # Very small heuristic fallback parser
#         out = {}
#         for f in features:
#             out[f] = None
#         # split by commas and look for "feature = value" patterns
#         tokens = [t.strip() for t in text.replace("\n", ",").split(",") if "=" in t]
#         for t in tokens:
#             try:
#                 k, v = t.split("=",1)
#                 k = k.strip().strip('"').strip("'")
#                 v = v.strip().strip('"').strip("'")
#                 if k in out:
#                     # try numeric
#                     try:
#                         out[k] = float(v) if "." in v else int(v)
#                     except:
#                         out[k] = v
#             except:
#                 continue
#         return out

import re
import numpy as np

def parse_query(query: str, feature_names: list):
    """
    Parses a natural-language or CSV-style query and returns
    a full-length feature vector. Missing values filled with 0.0.
    Works case-insensitively and matches partial names (e.g. 'glucose' matches 'Glucose').
    """
    query_lower = query.lower()
    # Map lowercase feature names to original ones
    name_map = {feat.lower(): feat for feat in feature_names}
    values = {}

    for feat_lower, feat_real in name_map.items():
        # Match e.g. "glucose=120" or "Glucose = 120"
        pattern = rf"{feat_lower}\s*=\s*([-+]?\d*\.?\d+)"
        match = re.search(pattern, query_lower)
        if match:
            values[feat_real] = float(match.group(1))

        # Allow short-form like "bmi=30" to match "BMI"
        short_feat = feat_lower.split("_")[0]
        pattern_short = rf"{short_feat}\s*=\s*([-+]?\d*\.?\d+)"
        match_short = re.search(pattern_short, query_lower)
        if match_short and feat_real not in values:
            values[feat_real] = float(match_short.group(1))

    if len(values) == 0:
        raise ValueError("No valid features found in input query.")

    # Fill missing features with 0 (mean-centered after scaling)
    complete_input = [values.get(feat, 0.0) for feat in feature_names]
    return np.array(complete_input).reshape(1, -1)

# def parse_query(query: str, feature_names: list):
#     """
#     Parses both natural language and structured key=value inputs.
#     Examples it supports:
#       - "area=2200, bedrooms=3"
#       - "predict house price for 3 bedrooms, 2 bathrooms, area 2200"
#       - "semi-furnished house with air conditioning"
#     """
#     query_lower = query.lower()
#     name_map = {feat.lower(): feat for feat in feature_names}
#     values = {}

#     # --- handle numeric features like "area 2200" or "area=2200"
#     for feat_lower, feat_real in name_map.items():
#         num_pattern = rf"{feat_lower}\s*[=:]?\s*([-+]?\d*\.?\d+)"
#         match = re.search(num_pattern, query_lower)
#         if match:
#             values[feat_real] = float(match.group(1))

#     # --- handle yes/no features
#     for feat_lower, feat_real in name_map.items():
#         if feat_lower in query_lower:
#             # e.g. "yes" or "no" after the keyword or alone
#             if re.search(rf"{feat_lower}.*yes", query_lower):
#                 values[feat_real] = 1.0
#             elif re.search(rf"{feat_lower}.*no", query_lower):
#                 values[feat_real] = 0.0

#     # --- handle furnishingstatus explicitly (categorical word mapping)
#     if "furnishingstatus" in name_map:
#         if "furnished" in query_lower:
#             if "semi" in query_lower:
#                 values[name_map["furnishingstatus"]] = 1.0  # semi-furnished
#             elif "unfurnished" in query_lower:
#                 values[name_map["furnishingstatus"]] = 0.0
#             else:
#                 values[name_map["furnishingstatus"]] = 2.0  # furnished

#     if len(values) == 0:
#         raise ValueError("No valid features found in input query.")

#     # Fill missing features with 0 (mean-centered)
#     complete_input = [values.get(f, 0.0) for f in feature_names]
#     return np.array(complete_input).reshape(1, -1)

def predict_with_llm(query, model, scaler, feature_names):
    """
    Dynamically handles both classification (with predict_proba)
    and regression (continuous) predictions.
    """
    X_input = parse_query(query, feature_names)
    X_scaled = scaler.transform(X_input)

    # Predict
    pred = model.predict(X_scaled)

    # Classification → has predict_proba()
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)
        conf = round(float(max(proba[0])), 3)
        return pred[0], conf

    # Regression → no predict_proba()
    else:
        return round(float(pred[0]), 3), None


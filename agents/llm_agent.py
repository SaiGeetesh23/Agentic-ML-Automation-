import re
import numpy as np

def parse_query(query: str, feature_names: list):
    """
    Parses a natural-language or CSV-style query and returns
    a full-length feature vector. Missing values filled with 0.0.
    Works case-insensitively and matches partial names (e.g. 'glucose' matches 'Glucose').
    """
    query_lower = query.lower()
    name_map = {feat.lower(): feat for feat in feature_names}
    values = {}
    for feat_lower, feat_real in name_map.items():
        pattern = rf"{feat_lower}\s*=\s*([-+]?\d*\.?\d+)"
        match = re.search(pattern, query_lower)
        if match:
            values[feat_real] = float(match.group(1))
        short_feat = feat_lower.split("_")[0]
        pattern_short = rf"{short_feat}\s*=\s*([-+]?\d*\.?\d+)"
        match_short = re.search(pattern_short, query_lower)
        if match_short and feat_real not in values:
            values[feat_real] = float(match_short.group(1))
    if len(values) == 0:
        raise ValueError("No valid features found in input query.")
    complete_input = [values.get(feat, 0.0) for feat in feature_names]
    return np.array(complete_input).reshape(1, -1)

def predict_with_llm(query, model, scaler, feature_names):
    """
    Dynamically handles both classification (with predict_proba)
    and regression (continuous) predictions.
    """
    X_input = parse_query(query, feature_names)
    X_scaled = scaler.transform(X_input)
    pred = model.predict(X_scaled)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)
        conf = round(float(max(proba[0])), 3)
        return pred[0], conf
    else:
        return round(float(pred[0]), 3), None


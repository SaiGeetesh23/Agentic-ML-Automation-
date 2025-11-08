# # agents/predictor_agent.py
# import os
# import joblib
# import uuid
# import numpy as np
# import json
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
# from model_factory import get_model
# from task_detector import detect_task_type_from_series


# class PredictorAgent:
#     def __init__(self, out_dir="artifacts/models"):
#         self.out_dir = out_dir
#         os.makedirs(self.out_dir, exist_ok=True)

#     def _load_npy_split(self, split_paths):
#         X_tr = np.load(split_paths["train"])
#         y_tr = np.load(split_paths["y_train"])
#         X_val = np.load(split_paths["val"])
#         y_val = np.load(split_paths["y_val"])
#         X_test = np.load(split_paths["test"])
#         y_test = np.load(split_paths["y_test"])
#         return (X_tr, y_tr), (X_val, y_val), (X_test, y_test)

#     def train(self, split_paths, features=None, model_name=None, task_type_hint=None, random_state=42):
#         """
#         Train a model automatically:
#          - detect task type if not provided,
#          - select model via model_factory,
#          - fit scaler + model,
#          - evaluate and save artifacts (model + scaler + metadata).
#         Returns metadata containing model path, metrics, task_type.
#         """
#         (X_tr, y_tr), (X_val, y_val), (X_test, y_test) = self._load_npy_split(split_paths)

#         import pandas as pd
#         from sklearn.model_selection import cross_val_score

#         # Detect task type
#         task_type = task_type_hint or detect_task_type_from_series(pd.Series(y_tr))
#         model = get_model(task_type, model_name=model_name, random_state=random_state)

#         # Fit scaler ONLY on training data
#         scaler = StandardScaler().fit(X_tr)
#         X_tr_s = scaler.transform(X_tr)
#         X_val_s = scaler.transform(X_val)
#         X_test_s = scaler.transform(X_test)

#         # Train model
#         model.fit(X_tr_s, y_tr)

#         # Evaluate on the **unseen test split**
#         if task_type == "classification":
#             preds = model.predict(X_test_s)
#             metrics = {
#                 "accuracy": float(accuracy_score(y_test, preds)),
#                 "f1_macro": float(f1_score(y_test, preds, average="macro")),
#             }

#             # sanity: print how many of each label were predicted
#             unique_pred, counts = np.unique(preds, return_counts=True)
#             print(f"[PredictorAgent] Prediction distribution: {dict(zip(unique_pred, counts))}")

#         else:
#             preds = model.predict(X_test_s)
#             metrics = {
#                 "rmse": float(mean_squared_error(y_test, preds, squared=False)),
#                 "r2": float(r2_score(y_test, preds)),
#             }

#         # Save model & scaler
#         run_id = str(uuid.uuid4())
#         model_path = os.path.join(self.out_dir, f"model_{run_id}.joblib")
#         scaler_path = os.path.join(self.out_dir, f"scaler_{run_id}.joblib")
#         joblib.dump(model, model_path)
#         joblib.dump(scaler, scaler_path)

#         # --- AUTO SAVE METADATA (feature names, task type, paths) ---
#         meta = {
#             "features": features,
#             "task_type": task_type,
#             "model_path": model_path,
#             "scaler_path": scaler_path,
#             "metrics": metrics,
#             "run_id": run_id,
#         }
#         meta_path = os.path.join(self.out_dir, f"metadata_{run_id}.json")
#         with open(meta_path, "w") as f:
#             json.dump(meta, f, indent=2)

#         # Save a shortcut to the latest metadata
#         latest_meta = os.path.join(self.out_dir, "latest_metadata.json")
#         with open(latest_meta, "w") as f:
#             json.dump(meta, f, indent=2)

#         print(f"[PredictorAgent] Saved metadata → {latest_meta}")
#         print(f"[PredictorAgent] Final Metrics: {metrics}")

#         return {
#             "task_type": task_type,
#             "model_path": model_path,
#             "scaler_path": scaler_path,
#             "features": features,
#             "metrics": metrics,
#         }


#     def load_latest_metadata(self):
#         """Load latest metadata (features, paths, etc.) if available."""
#         meta_path = os.path.join(self.out_dir, "latest_metadata.json")
#         if os.path.exists(meta_path):
#             with open(meta_path, "r") as f:
#                 return json.load(f)
#         else:
#             print("[PredictorAgent] No latest_metadata.json found.")
#             return None

#     def load_model(self, model_path, scaler_path):
#         model = joblib.load(model_path)
#         scaler = joblib.load(scaler_path)
#         return model, scaler

#     def predict_row(self, model, scaler, feature_row):
#         """
#         feature_row: list or 1D numpy array of feature values in same order as during training.
#         Returns the model's prediction (class label or value).
#         """
#         x = np.array(feature_row, dtype=float).reshape(1, -1)
#         x_s = scaler.transform(x)
#         pred = model.predict(x_s)
#         return pred[0]

#     def predict_batch(self, model, scaler, X):
#         X = np.array(X, dtype=float)
#         X_s = scaler.transform(X)
#         return model.predict(X_s)

import joblib, json, os, uuid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datetime import datetime

def train_and_evaluate(X_train, X_val, y_train, y_val, scaler, feature_names=None, dataset_name=None):
    import joblib, json, os, uuid
    from datetime import datetime
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import accuracy_score, f1_score, classification_report, mean_squared_error, r2_score
    import numpy as np

    #  Detect task type automatically
    task_type = "classification" if len(np.unique(y_train)) <= 10 else "regression"

    if task_type == "classification":
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        metrics = {
            "task_type": task_type,
            "accuracy": accuracy_score(y_val, preds),
            "f1_score": f1_score(y_val, preds, average="weighted"),
            "classification_report": classification_report(y_val, preds, output_dict=True)
        }
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        metrics = {
            "task_type": task_type,
            "rmse": float(np.sqrt(mean_squared_error(y_val, preds))),
            "r2_score": r2_score(y_val, preds)
        }

    #  Save metadata
    uid = str(uuid.uuid4())
    os.makedirs("artifacts/models", exist_ok=True)
    model_path = f"artifacts/models/model_{uid}.joblib"
    scaler_path = f"artifacts/models/scaler_{uid}.joblib"
    metadata_path = f"artifacts/models/metadata_{uid}.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "uuid": uid,
        "timestamp": datetime.now().isoformat(),
        "dataset_name": dataset_name or "unknown_dataset",
        "model_type": type(model).__name__,
        "task_type": task_type,
        "metrics": metrics,
        "paths": {"model": model_path, "scaler": scaler_path},
        "feature_names": feature_names
    }

    json.dump(metadata, open(metadata_path, "w"), indent=4)
    json.dump(metadata, open("artifacts/models/latest_metadata.json", "w"), indent=4)

    print(f" Model training complete ({task_type}).")
    print(json.dumps(metrics, indent=4))
    return model, metrics

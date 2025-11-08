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

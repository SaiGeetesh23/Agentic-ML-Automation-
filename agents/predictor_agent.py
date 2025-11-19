import os
import uuid
import json
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
class PredictorAgent:
    """
    Universal Predictor Agent
    - Trains model based on task type (classification/regression)
    - Saves model
    - Provides .predict() for prediction
    """

    def __init__(self):
        pass

    def train(
        self,
        X_train,
        X_val,
        y_train,
        y_val,
        scaler,
        feature_names,
        dataset_name,
        task_type,
    ):
        if task_type == "classification":
            model = LogisticRegression(
                max_iter=500,
                class_weight="balanced"
            )
        else:
            model = LinearRegression()
        model.fit(X_train, y_train)  
        if task_type == "classification":
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average="weighted")
            report = classification_report(y_val, y_pred, output_dict=True)
            metrics = {
                "task_type": "classification",
                "accuracy": acc,
                "f1_score_weighted": f1,
                "classification_report": report,
            }

        else: 
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            metrics = {
                "task_type": "regression",
                "rmse": rmse,
                "r2_score": r2,
            }   
        os.makedirs("artifacts/models", exist_ok=True)
        model_uid = str(uuid.uuid4())
        model_path = f"artifacts/models/model_{model_uid}.joblib"
        joblib.dump(model, model_path)
        print(f"Model training complete. Saved model to: {model_path}")
        return model, metrics, model_path

    def predict(self, model_path, metadata, prepared_input_row):
        """
        - model_path: saved model file
        - metadata: dict containing feature names & scaler path
        - prepared_input_row: already scaled row from DataAgent.prepare_input()
        """
        model = joblib.load(model_path)
        y_pred = model.predict(prepared_input_row)[0]
        if hasattr(model, "predict_proba"):
            confidence = float(
                np.max(model.predict_proba(prepared_input_row))
            )
        else:
            confidence = 1.0
        return {
            "prediction": float(y_pred),
            "confidence": confidence,
        }

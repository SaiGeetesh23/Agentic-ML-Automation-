import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score

class AnalysisAgent:
    """
    Universal Analysis Agent
    - Generates confusion matrix (classification)
    - Generates regression fit plot
    - Generates correlation heatmap
    """

    def __init__(self):
        pass

    def _load_latest_artifacts(self):
        model_files = sorted(os.listdir("artifacts/models"))
        latest_model_file = model_files[-1]
        model_path = f"artifacts/models/{latest_model_file}"
        model = joblib.load(model_path)   
        data_files = sorted(os.listdir("artifacts/data"))
        meta_files = [m for m in data_files if m.startswith("metadata")]
        latest_meta_file = meta_files[-1]
        with open(f"artifacts/data/{latest_meta_file}", "r") as f:
            metadata = json.load(f)   
        test_files = [m for m in data_files if m.startswith("test_data")]
        latest_test_file = test_files[-1]
        test_data = joblib.load(f"artifacts/data/{latest_test_file}")
        X_test = test_data["X_test"]
        y_test = test_data["y_test"]
        return model, metadata, X_test, y_test

    def run_analysis(self):
        model, metadata, X_test, y_test = self._load_latest_artifacts()
        task_type = metadata["task_type"]
        feature_names = metadata["feature_names"]
        os.makedirs("artifacts/analysis", exist_ok=True)
        if task_type == "regression":
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, alpha=0.6)
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Regression Fit")
            plt.grid(True)
            plt.savefig("artifacts/analysis/regression_fit.png")
            plt.close()
            print("Regression fit plot saved.")
            print(f"RMSE = {rmse:.4f}, R^2 = {r2:.4f}")
            return {
                "task_type": "regression",
                "rmse": rmse,
                "r2_score": r2,
            }
        else:
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.savefig("artifacts/analysis/confusion_matrix.png")
            plt.close()
            print("Confusion matrix saved.")
            return {
                "task_type": "classification",
                "confusion_matrix": cm.tolist(),
            } 
    def correlation_plot(self, df):
        os.makedirs("artifacts/analysis", exist_ok=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig("artifacts/analysis/correlation_heatmap.png")
        plt.close()
        print("Correlation heatmap saved.")

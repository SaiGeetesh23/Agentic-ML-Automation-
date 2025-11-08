import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, r2_score

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Save confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_loss_curve(y_true, y_pred, save_path):
    """Save regression error plot (predicted vs actual)"""
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Regression Prediction Fit")
    plt.grid(True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def generate_analysis():
    """Main function: generates and saves confusion matrix or regression curve"""
    metadata_path = "artifacts/models/latest_metadata.json"
    if not os.path.exists(metadata_path):
        print(" No trained model metadata found. Run orchestrator_predict.py first.")
        return
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    task_type = metadata.get("task_type", "classification")
    model_path = metadata["paths"]["model"]
    scaler_path = metadata["paths"]["scaler"]
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    test_data_path = "artifacts/data/test_data.joblib"
    if not os.path.exists(test_data_path):
        print(" No saved test data found. Please save test split in data_agent if needed.")
        return
    X_test, y_test = joblib.load(test_data_path)
    import pandas as pd
    X_test_df = pd.DataFrame(X_test, columns=metadata["feature_names"])
    X_test_scaled = scaler.transform(X_test_df)
    y_pred = model.predict(X_test_scaled)
    os.makedirs("artifacts/analysis", exist_ok=True)
    if task_type == "classification":
        cm_path = "artifacts/analysis/confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred, cm_path)
        print(f" Confusion matrix saved to {cm_path}")
    else:
        loss_path = "artifacts/analysis/regression_fit.png"
        plot_loss_curve(y_test, y_pred, loss_path)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f" Regression analysis saved to {loss_path}")
        print(f" RMSE = {rmse:.3f}, R² = {r2:.3f}")

if __name__ == "__main__":
    generate_analysis()

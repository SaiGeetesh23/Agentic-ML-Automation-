# import torch, torch.nn as nn, torch.optim as optim
# import numpy as np
# import uuid
# import os
# from models.simple_mlp import SimpleMLP
# from utils.viz import save_confusion_matrix, save_metric_curve
# from sklearn.metrics import accuracy_score, f1_score, classification_report

# class AnalysisAgent:
#     def __init__(self, out_dir="artifacts"):
#         self.out_dir = out_dir
#         os.makedirs(out_dir, exist_ok=True)

#     def _load_data(self, paths):
#         X_tr = np.load(paths["train"])
#         y_tr = np.load(paths["y_train"])
#         X_val = np.load(paths["val"])
#         y_val = np.load(paths["y_val"])
#         X_test = np.load(paths["test"])
#         y_test = np.load(paths["y_test"])
#         return (X_tr,y_tr),(X_val,y_val),(X_test,y_test)

#     def run(self, split_paths, config=None):
#         cfg = config or {}
#         (X_tr,y_tr),(X_val,y_val),(X_test,y_test) = self._load_data(split_paths)
#         input_dim = X_tr.shape[1]
#         num_classes = int(max(np.unique(y_tr)) + 1)

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model = SimpleMLP(input_dim=input_dim, hidden=cfg.get("hidden",64), num_classes=num_classes).to(device)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
#         epochs = cfg.get("epochs", 20)
#         batch = cfg.get("batch", 64)

#         train_losses, val_losses = [], []
#         for ep in range(1, epochs+1):
#             model.train()
#             perm = np.random.permutation(len(X_tr))
#             losses = []
#             for i in range(0, len(X_tr), batch):
#                 idx = perm[i:i+batch]
#                 xb = torch.tensor(X_tr[idx], dtype=torch.float32).to(device)
#                 yb = torch.tensor(y_tr[idx], dtype=torch.long).to(device)
#                 optimizer.zero_grad()
#                 out = model(xb)
#                 loss = criterion(out, yb)
#                 loss.backward()
#                 optimizer.step()
#                 losses.append(loss.item())
#             train_losses.append(np.mean(losses))

#             model.eval()
#             with torch.no_grad():
#                 xb = torch.tensor(X_val, dtype=torch.float32).to(device)
#                 loss_val = criterion(model(xb), torch.tensor(y_val, dtype=torch.long).to(device)).item()
#                 val_losses.append(loss_val)

#         model.eval()
#         with torch.no_grad():
#             xb = torch.tensor(X_test, dtype=torch.float32).to(device)
#             out = model(xb).cpu().numpy()
#             preds = out.argmax(axis=1)
#         acc = accuracy_score(y_test, preds)
#         f1 = f1_score(y_test, preds, average='macro')
#         report = classification_report(y_test, preds, output_dict=True)

#         run_id = str(uuid.uuid4())
#         model_path = os.path.join(self.out_dir, f"model_{run_id}.pt")
#         torch.save(model.state_dict(), model_path)
#         cm_path = os.path.join(self.out_dir, f"cm_{run_id}.html")
#         save_confusion_matrix(y_test, preds, labels=list(np.unique(y_test).tolist()), out_html=cm_path)
#         loss_path = os.path.join(self.out_dir, f"loss_{run_id}.html")
#         save_metric_curve(train_losses, val_losses, loss_path, metric_name="loss")

#         return {
#             "metrics": {"accuracy": acc, "f1_macro": f1},
#             "model_path": model_path,
#             "report": report,
#             "plots": {"confusion_matrix": cm_path, "loss_curve": loss_path}
#         }


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

    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Load test data if available
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

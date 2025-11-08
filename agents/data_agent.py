from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib, os, uuid, json

class DataAgent:
    def __init__(self, dataset_path, target_col):
        self.dataset_path = dataset_path
        self.target_col = target_col

    def execute(self):
        df = pd.read_csv(self.dataset_path)
        print(f"📊 Loaded dataset with shape: {df.shape}")

        # Drop any irrelevant ID columns automatically
        if "id" in df.columns:
            df = df.drop(columns=["id"])

        # Separate features and target
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # Detect categorical columns and encode them
        cat_cols = X.select_dtypes(include=["object"]).columns
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Scale numeric features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        #  Detect task type (classification or regression)
        task_type = "classification" if len(np.unique(y)) <= 10 else "regression"
        print(f"🔍 Detected task type: {task_type}")

        #  For classification → stratify
        #  For regression → normal random split
        if task_type == "classification":
            stratify_arg = y
        else:
            stratify_arg = None

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=stratify_arg
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=None
        )

        # Save artifacts
        uid = str(uuid.uuid4())
        os.makedirs("artifacts/data", exist_ok=True)
        joblib.dump(scaler, f"artifacts/data/scaler_{uid}.joblib")
        joblib.dump((X_test, y_test), "artifacts/data/test_data.joblib")

        print(f" Preprocessing complete. Task={task_type}, Train={X_train.shape}, Test={X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test, scaler, None

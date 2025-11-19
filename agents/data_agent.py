import uuid
import os
import json
import csv
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataAgent:
    """
    Final robust DataAgent:
    - auto-detect delimiter
    - handle european decimals and -200 sentinel
    - coerce boolean-like strings
    - fill numeric NaNs with median, categorical with mode/'unknown'
    - encode categorical features with LabelEncoder
    - compute feature_means AFTER encoding (numeric)
    - save scaler, encoders, metadata, test split
    - prepare_input handles categorical mapping from user input
    """

    def __init__(self, dataset_path=None, target_col=None):
        self.dataset_path = dataset_path
        self.target_col = target_col

    def _detect_delimiter(self, path):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                sample = "".join([f.readline() for _ in range(5)])
                dialect = csv.Sniffer().sniff(sample, delimiters=",;")
                return dialect.delimiter
        except Exception:
            return ","

    def _drop_id_columns(self, df):
        drop_cols = [c for c in df.columns if c is not None and c.lower() in ["id", "unnamed: 0", "", " "]]
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")
        return df

    def _coerce_boolean_like(self, s: pd.Series):
        
        if s.dtype == object or s.dtype.name == "category":
            lowered = s.astype(str).str.strip().str.lower()
            mapping = {
                "yes": 1, "no": 0,
                "y": 1, "n": 0,
                "true": 1, "false": 0,
                "t": 1, "f": 0,
                "1": 1, "0": 0
            }
            
            matches = lowered.isin(mapping.keys()).sum()
            if matches / max(1, len(lowered)) > 0.3:
                return lowered.map(mapping).astype(float)
        return s

    def _clean_target(self, df):
        """Return (df_filtered, y_array, target_encoder or None)"""
        y = df[self.target_col].astype(str)
        y = y.replace(["nan", "NaN", "None", "?", "", " "], np.nan)
        valid_mask = ~y.isna()
        df = df[valid_mask].reset_index(drop=True)
        y = y[valid_mask]
        y_num = pd.to_numeric(y, errors="coerce")
        if y_num.notna().sum() > 0 and y_num.isna().sum() == 0:
            return df, y_num.values, None 
        try:
            le = LabelEncoder()
            y_enc = le.fit_transform(y.astype(str))
            return df, y_enc, le
        except Exception:     
            y_filled = pd.to_numeric(y, errors="coerce")
            y_filled = pd.Series(y_filled).fillna(y_filled.median()).values
            return df, y_filled, None

    def execute(self):
        if not self.dataset_path or not self.target_col:
            raise ValueError("dataset_path and target_col must be provided")      
        delimiter = self._detect_delimiter(self.dataset_path)
        df = pd.read_csv(self.dataset_path, sep=delimiter, encoding="utf-8", engine="python")
        print(f"Loaded dataset: {self.dataset_path} with shape {df.shape}")
        df = self._drop_id_columns(df)      
        df = df.replace(-200, np.nan)
        df = df.replace(",", ".", regex=True)    
        df = df.dropna(axis=1, how="all")
        if self.target_col not in df.columns:
            raise ValueError(f" Target column '{self.target_col}' not found in dataset")   
        df, y, target_encoder = self._clean_target(df)  
        X = df.drop(columns=[self.target_col]).copy()  
        for col in X.columns:
            X[col] = self._coerce_boolean_like(X[col])
        X = X.apply(pd.to_numeric, errors="ignore")
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 0:
            X[num_cols] = X[num_cols].apply(lambda c: c.fillna(c.median()))
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in cat_cols:
            try:
                mode = X[col].mode(dropna=True)
                fill_val = mode.iloc[0] if not mode.empty else "unknown"
                X[col] = X[col].fillna(fill_val)
            except Exception:
                X[col] = X[col].fillna("unknown")
        encoders = {}
        for col in cat_cols:
            if X[col].dtype == "object" or X[col].dtype.name == "category":
                try:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    encoders[col] = le
                    print(f"Encoded feature column: {col}")
                except Exception:       
                    uniques = sorted(X[col].astype(str).unique())
                    mapping = {v: i for i, v in enumerate(uniques)}
                    X[col] = X[col].astype(str).map(mapping).astype(float)               
        unique_y = np.unique(y)
        is_classification = (np.issubdtype(y.dtype, np.integer) and len(unique_y) <= 20)
        task_type = "classification" if is_classification else "regression"
        print(f"Detected task type: {task_type} ({len(unique_y)} labels)")
        feature_means = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):    
                feature_means[col] = float(pd.to_numeric(X[col], errors="coerce").median())
            else:               
                try:
                    m = X[col].mode(dropna=True)
                    feature_means[col] = m.iloc[0] if not m.empty else 0.0
                except Exception:
                    feature_means[col] = 0.0       
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)  
        stratify = y if task_type == "classification" else None
        X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=stratify)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  
        uid = str(uuid.uuid4())
        os.makedirs("artifacts/data", exist_ok=True)
        os.makedirs("artifacts/models", exist_ok=True)
        scaler_path = f"artifacts/data/scaler_{uid}.joblib"
        joblib.dump(scaler, scaler_path)
        encoders_path = None
        if encoders:
            encoders_path = f"artifacts/data/encoders_{uid}.joblib"
            joblib.dump(encoders, encoders_path)
        test_data_path = f"artifacts/data/test_data_{uid}.joblib"
        joblib.dump({"X_test": X_test, "y_test": y_test, "feature_names": X.columns.tolist()}, test_data_path)
        metadata = {
            "uid": uid,
            "dataset_path": self.dataset_path,
            "target_col": self.target_col,
            "task_type": task_type,
            "feature_names": X.columns.tolist(),
            "scaler_path": scaler_path,
            "encoders_path": encoders_path,
            "is_categorical_target": bool(target_encoder),
            "feature_means": feature_means,
        }
        meta_path = f"artifacts/data/metadata_{uid}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Preprocessing complete. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test, scaler, metadata
    
    def prepare_input(self, feature_values: dict, metadata: dict):
        """
        Build a scaled input row for prediction.
        - Missing features are filled with dataset medians (feature_means)
        - Categorical features are encoded using saved encoders
        - Accepts user supplying either raw label (e.g., 'm') or numeric code (e.g., 1)
        """  
        scaler = joblib.load(metadata["scaler_path"])
        feature_names = metadata["feature_names"]
        means = metadata.get("feature_means", {})
        encoders_path = metadata.get("encoders_path", None)
        encoders = {}
        if encoders_path and os.path.exists(encoders_path):
            try:
                encoders = joblib.load(encoders_path)
            except Exception:
                encoders = {}
        row = []
        for fname in feature_names:
            if fname in feature_values:
                raw_val = feature_values[fname]        
                if fname in encoders:
                    le = encoders[fname]          
                    try:
                        encoded = le.transform([str(raw_val)])[0]
                        row.append(float(encoded))
                        continue
                    except Exception:             
                        try:
                            intv = int(raw_val)
                            if 0 <= intv < len(le.classes_):
                                row.append(float(intv))
                                continue
                        except Exception:
                            pass       
                        try:
                            mode = le.transform([str(le.classes_[0])])[0]
                            row.append(float(mode))
                            continue
                        except Exception:
                            row.append(float(means.get(fname, 0.0)))
                            continue
                else:          
                    try:
                        row.append(float(raw_val))
                        continue
                    except Exception:          
                        s = str(raw_val).strip().lower()
                        if s in {"yes","y","true","t","1"}:
                            row.append(1.0); continue
                        if s in {"no","n","false","f","0"}:
                            row.append(0.0); continue                       
                        row.append(float(means.get(fname, 0.0)))
                        continue
            else:            
                mv = means.get(fname, 0.0)
                try:
                    row.append(float(mv))
                except Exception:                  
                    if fname in encoders:
                        le = encoders[fname]
                        try:
                            enc_mode = le.transform([str(le.classes_[0])])[0]
                            row.append(float(enc_mode))
                        except Exception:
                            row.append(0.0)
                    else:
                        row.append(0.0)
        row_scaled = scaler.transform([row])
        return row_scaled

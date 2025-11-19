import os
import json
import joblib
from langchain_openai import ChatOpenAI


class ReviewerAgent:
    """
    Universal ReviewerAgent
    - Reads metadata, model, metrics
    - Generates professional ML review using GPT-4o-mini
    """

    def __init__(self, api_key):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=api_key
        )

    
    def _load_latest(self): 
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

        return model, metadata, X_test, y_test, model_path
    
    def generate_review(self):
        (
            model,
            metadata,
            X_test,
            y_test,
            model_path
        ) = self._load_latest()
        dataset_name = metadata["dataset_path"]
        target = metadata["target_col"]
        task_type = metadata["task_type"]
        features = metadata["feature_names"]
        review_prompt = f"""
You are a machine learning model reviewer.

Review the following trained model in a structured, professional manner.

Dataset: {dataset_name}
Target Column: {target}
Task Type: {task_type}
Features Used: {features}
Model File: {model_path}

Evaluate and describe:

1. Model Strengths
2. Model Weaknesses
3. Overfitting/Underfitting analysis
4. Class imbalance issues (if classification)
5. Quality of feature selection
6. Suggestions for better algorithms
7. Suggestions for better preprocessing
8. Hyperparameter tuning ideas
9. Final Recommendation

Write the review clearly and professionally.
"""
        result = self.llm.invoke(review_prompt).content
        os.makedirs("artifacts/reviews", exist_ok=True)
        review_path = "artifacts/reviews/review_latest.txt"
        with open(review_path, "w") as f:
            f.write(result)
        print("Review saved to artifacts/reviews/review_latest.txt")
        return result

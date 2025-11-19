import os
import json
import joblib
from langchain.tools import tool
from agents.data_agent import DataAgent
from agents.predictor_agent import PredictorAgent
from agents.reviewer_agent import ReviewerAgent
from agents.analysis_agent import AnalysisAgent
from agents.llm_agent import LLMAgent

def session_file():
    """Return session file path that stores current dataset & target."""
    return "artifacts/session/latest_dataset.json"


def load_session_dataset():
    """Load the dataset path and target column selected in the current session."""
    if not os.path.exists(session_file()):
        raise ValueError("No dataset loaded. Run: load dataset <path> target <col>")
    return json.load(open(session_file(), "r"))


def load_latest_metadata():
    """Load the latest metadata file produced by DataAgent."""
    files = sorted(os.listdir("artifacts/data"))
    meta_files = [f for f in files if f.startswith("metadata")]
    if not meta_files:
        raise ValueError("No metadata found. Run DataAgent first.")
    return json.load(open(f"artifacts/data/{meta_files[-1]}", "r"))


def load_latest_model():
    """Return the newest trained model path from artifacts/models."""
    model_files = sorted(os.listdir("artifacts/models"))
    if not model_files:
        raise ValueError("No trained model found. Run: train model")
    return f"artifacts/models/{model_files[-1]}"


@tool("run_data_agent", return_direct=True)
def run_data_agent_tool(dataset_path: str, target_col: str):
    """
    Preprocess the dataset using DataAgent.
    Saves session metadata so all other tools use the same dataset.

    Args:
        dataset_path: Path to the dataset (CSV file).
        target_col: Name of the target column.
    """

    os.makedirs("artifacts/session", exist_ok=True)
    json.dump(
        {"dataset_path": dataset_path, "target_col": target_col},
        open(session_file(), "w")
    )  
    da = DataAgent(dataset_path, target_col)
    da.execute()
    return f"Data loaded and processed for dataset={dataset_path}, target={target_col}"

@tool("train_model", return_direct=True)
def train_model_tool():
    """
    Train a machine learning model using the dataset chosen in this session.
    Uses DataAgent for preprocessing and PredictorAgent for training.
    """
    cfg = load_session_dataset()
    dataset_path = cfg["dataset_path"]
    target_col = cfg["target_col"]
    da = DataAgent(dataset_path, target_col)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, metadata = da.execute()
    pa = PredictorAgent()
    model, metrics, model_path = pa.train(
        X_train,
        X_val,
        y_train,
        y_val,
        scaler,
        metadata["feature_names"],
        metadata["dataset_path"],
        metadata["task_type"]
    )
    return {
        "status": " Model trained successfully",
        "model_path": model_path,
        "metrics": metrics
    }

@tool("predict_values", return_direct=True)
def predict_values_tool(user_text: str):
    """
    Predict target value using natural-language input.
    Missing features are automatically filled using dataset medians.

    Args:
        user_text: Example -> "glucose=120 bmi=32 age=45"
    """
    cfg = load_session_dataset()
    dataset_path = cfg["dataset_path"]
    target_col = cfg["target_col"]
    metadata = load_latest_metadata()
    model_path = load_latest_model()
    llm = LLMAgent(api_key=os.getenv("OPENAI_API_KEY"))
    parsed = llm.parse_features(user_text)

    # normalized = {}
    # for user_key, val in parsed.items():
    #     for fname in metadata["feature_names"]:
    #         if user_key.lower() == fname.lower():
    #             normalized[fname] = val
    #             break

    # parsed = normalized
    normalized = {}
    feature_map = {fname.lower().strip(): fname for fname in metadata["feature_names"]}
    for user_key, val in parsed.items():
        key = user_key.lower().strip()
        if key in feature_map:
            normalized[feature_map[key]] = val
    parsed = normalized
    da = DataAgent(dataset_path, target_col)
    prepared_row = da.prepare_input(parsed, metadata)  
    pa = PredictorAgent()
    result = pa.predict(model_path, metadata, prepared_row)
    return {
        "parsed_features": parsed,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "task_type": metadata["task_type"]
    }

@tool("run_analysis", return_direct=True)
def run_analysis_tool():
    """
    Run AnalysisAgent to generate confusion matrix or regression fit plot.
    Output saved under artifacts/analysis/.
    """
    load_session_dataset()
    aa = AnalysisAgent()
    result = aa.run_analysis()
    return {
        "status": " Analysis completed",
        "result": result
    }

@tool("generate_review", return_direct=True)
def generate_review_tool():
    """
    Generate a professional machine learning model review using ReviewerAgent.
    Saves the review text into artifacts/reviews/review_latest.txt.
    """
    load_session_dataset()
    reviewer = ReviewerAgent(api_key=os.getenv("OPENAI_API_KEY"))
    review = reviewer.generate_review()
    return {
        "status": " Review generated successfully",
        "review_text": review
    }

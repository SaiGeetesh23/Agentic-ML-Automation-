import os
import json
import joblib
from agents.llm_agent import predict_with_llm

def load_latest_model():
    metadata_path = "artifacts/models/latest_metadata.json"
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("No trained model found. Please run orchestrator_predict.py first.")  
    metadata = json.load(open(metadata_path))
    model = joblib.load(metadata["paths"]["model"])
    scaler = joblib.load(metadata["paths"]["scaler"])
    feature_names = metadata.get("feature_names")
    task_type = metadata.get("task_type", "classification")
    dataset_name = metadata.get("dataset_name", "unknown_dataset")
    return model, scaler, feature_names, metadata, task_type, dataset_name

def main():
    print("\n Loading latest trained model...")
    model, scaler, feature_names, metadata, task_type, dataset_name = load_latest_model()
    print(f" Model loaded ({task_type}) for dataset: {dataset_name}")
    print(f" Features: {len(feature_names)} | {', '.join(feature_names[:8])}...")
    print("\nYou can enter feature=value pairs or natural queries.")
    print("Example:")
    print("  Glucose=120, BMI=30, Age=40")
    print("Type 'exit' to quit.\n")
    while True:
        query = input("Enter values (or natural query): ").strip()
        if query.lower() in ["exit", "quit"]:
            print(" Exiting CLI.")
            break
        if not query:
            continue
        try:
            pred, conf = predict_with_llm(query, model, scaler, feature_names)
            if task_type == "classification":
                print(f" Prediction: Class_{int(pred)} (Confidence={conf})")
            else:
                print(f" Predicted Value: {pred:.4f}")
        except Exception as e:
            print(" Error:", e)
            print("Try again with feature=value format (e.g., Glucose=120, BMI=30).")

if __name__ == "__main__":
    main()

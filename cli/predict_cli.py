# # # cli/predict_cli.py
# # import os
# # import json
# # import argparse
# # import joblib
# # from agents.predictor_agent import PredictorAgent
# # from agents.llm_agent import LLMQueryAgent

# # def load_latest_model_dir(models_dir="artifacts/models"):
# #     """
# #     Finds the most recent model+scaler pair saved by PredictorAgent (by timestamp).
# #     Assumes model_*.joblib and scaler_*.joblib exist.
# #     Returns paths (model_path, scaler_path) or raises.
# #     """
# #     import glob
# #     models = sorted(glob.glob(os.path.join(models_dir, "model_*.joblib")), key=os.path.getmtime, reverse=True)
# #     scalers = sorted(glob.glob(os.path.join(models_dir, "scaler_*.joblib")), key=os.path.getmtime, reverse=True)
# #     if not models or not scalers:
# #         raise FileNotFoundError("No saved models/scalers found in " + models_dir)
# #     return models[0], scalers[0]

# # def run_cli(features, model_path, scaler_path, llm_agent=None):
# #     pa = PredictorAgent()
# #     model, scaler = pa.load_model(model_path, scaler_path)
# #     print("Model loaded:", model_path)
# #     print("Enter 'exit' to quit.")
# #     print("You can either:")
# #     print("- type comma-separated values in the exact feature order:")
# #     print("   e.g. 5.1,3.5,1.4,0.2")
# #     print("- OR type a natural-language request (if LLM available):")
# #     print("   e.g. predict diabetes for glucose=140, bmi=28, age=45")

# #     while True:
# #         inp = input(">> ").strip()
# #         if inp.lower() in ("exit", "quit"):
# #             break
# #         # if it looks like comma-separated numbers -> direct predict
# #         if "," in inp and all(token.strip().replace(".","",1).lstrip("-").isdigit() for token in inp.split(",")):
# #             vals = [float(x.strip()) for x in inp.split(",")]
# #             pred = pa.predict_row(model, scaler, vals)
# #             print("Prediction:", pred)
# #             continue

# #         # otherwise try LLM parse (if agent available)
# #         if llm_agent is not None:
# #             parsed = llm_agent.parse_query(features, inp)
# #             # construct feature vector in order
# #             vec = []
# #             for f in features:
# #                 v = parsed.get(f, None)
# #                 if v is None:
# #                     print(f"Feature '{f}' missing in the user input. Please provide all features or use comma input.")
# #                     vec = None
# #                     break
# #                 vec.append(v)
# #             if vec is None:
# #                 continue
# #             # ensure numeric conversion where possible
# #             vec = [float(x) for x in vec]
# #             pred = pa.predict_row(model, scaler, vec)
# #             print("Prediction:", pred)
# #             continue

# #         print("Couldn't parse input. Provide CSV row or enable LLM parsing.")

# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--models-dir", default="artifacts/models")
# #     parser.add_argument("--features", type=str, help="JSON list of features in order", required=False)
# #     args = parser.parse_args()

# #     model_path, scaler_path = load_latest_model_dir(args.models_dir)

# #     # If user provided features via CLI, use them; otherwise attempt to read from a saved metadata file
# #     features = None
# #     meta_path = os.path.join(args.models_dir, "latest_metadata.json")
# #     if args.features:
# #         features = json.loads(args.features)
# #     elif os.path.exists(meta_path):
# #         features = json.load(open(meta_path, "r")).get("features")
# #     else:
# #         # As a last resort, ask the user
# #         features = input("Enter comma-separated feature names (in order): ").strip().split(",")

# #     # attempt to init LLM agent (if API key present)
# #     try:
# #         llm_agent = LLMQueryAgent()
# #         print("LLM parser available.")
# #     except Exception:
# #         llm_agent = None
# #         print("LLM parser not available (set OPENAI_API_KEY to enable).")

# #     run_cli(features, model_path, scaler_path, llm_agent=llm_agent)


# # cli/predict_cli.py
# import os
# import json
# import argparse
# import joblib
# import numpy as np
# from agents.predictor_agent import PredictorAgent
# from agents.llm_agent import LLMQueryAgent


# def load_latest_model_dir(models_dir="artifacts/models"):
#     """
#     Finds the most recent model+scaler pair saved by PredictorAgent (by timestamp).
#     Assumes model_*.joblib and scaler_*.joblib exist.
#     Returns paths (model_path, scaler_path).
#     """
#     import glob
#     models = sorted(
#         glob.glob(os.path.join(models_dir, "model_*.joblib")),
#         key=os.path.getmtime, reverse=True
#     )
#     scalers = sorted(
#         glob.glob(os.path.join(models_dir, "scaler_*.joblib")),
#         key=os.path.getmtime, reverse=True
#     )
#     if not models or not scalers:
#         raise FileNotFoundError(f"No saved models/scalers found in {models_dir}")
#     return models[0], scalers[0]


# def run_cli(features, model_path, scaler_path, llm_agent=None):
#     pa = PredictorAgent()
#     model, scaler = pa.load_model(model_path, scaler_path)
#     print("Model loaded:", model_path)
#     print("Enter 'exit' to quit.")
#     print("You can either:")
#     print("- type comma-separated values in the exact feature order:")
#     print("   e.g. 5.1,3.5,1.4,0.2")
#     print("- OR type a natural-language request (if LLM available):")
#     print("   e.g. predict breast cancer for radius_mean=15, texture_mean=18, area_mean=650")

#     while True:
#         inp = input(">> ").strip()
#         if inp.lower() in ("exit", "quit"):
#             break

#         # --- direct numeric input ---
#         if "," in inp and all(
#             token.strip().replace(".", "", 1).lstrip("-").isdigit()
#             for token in inp.split(",")
#         ):
#             vals = np.array([float(x.strip()) for x in inp.split(",")], dtype=float)

#             # ---- Auto-center user inputs (Option 1) ----
#             means = getattr(scaler, "mean_", None)
#             scales = getattr(scaler, "scale_", None)
#             if means is not None and scales is not None and len(vals) == len(means):
#                 z = (vals - means) / scales
#                 z = np.clip(z, -2, 2)           # keep within [-2σ, +2σ]
#                 vals = means + z * scales

#             pred = pa.predict_row(model, scaler, vals)

#             # Print with confidence if possible
#             if hasattr(model, "predict_proba"):
#                 proba = model.predict_proba(scaler.transform([vals]))[0]
#                 conf = float(np.max(proba))
#                 print(f"Prediction: {int(pred)} (confidence malignant={proba[1]:.2f}, overall={conf:.2f})")
#             else:
#                 print("Prediction:", int(pred))
#             continue

#         # --- LLM-based natural language parsing ---
#         if llm_agent is not None:
#             parsed = llm_agent.parse_query(features, inp)
#             vec = []
#             for f in features:
#                 v = parsed.get(f, None)
#                 if v is None:
#                     print(f"Feature '{f}' missing in user input. Provide all features or use comma input.")
#                     vec = None
#                     break
#                 vec.append(v)
#             if vec is None:
#                 continue
#             vec = [float(x) for x in vec]

#             vals = np.array(vec, dtype=float)

#             # Auto-center for LLM-parsed values too
#             means = getattr(scaler, "mean_", None)
#             scales = getattr(scaler, "scale_", None)
#             if means is not None and scales is not None and len(vals) == len(means):
#                 z = (vals - means) / scales
#                 z = np.clip(z, -2, 2)
#                 vals = means + z * scales

#             pred = pa.predict_row(model, scaler, vals)
#             if hasattr(model, "predict_proba"):
#                 proba = model.predict_proba(scaler.transform([vals]))[0]
#                 conf = float(np.max(proba))
#                 print(f"Prediction: {int(pred)} (confidence malignant={proba[1]:.2f}, overall={conf:.2f})")
#             else:
#                 print("Prediction:", int(pred))
#             continue

#         print("Couldn't parse input. Provide comma-separated numbers or enable LLM parsing.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--models-dir", default="artifacts/models")
#     parser.add_argument("--features", type=str, help="JSON list of features in order", required=False)
#     args = parser.parse_args()

#     model_path, scaler_path = load_latest_model_dir(args.models_dir)

#     # Load feature order from metadata or user
#     features = None
#     meta_path = os.path.join(args.models_dir, "latest_metadata.json")
#     if args.features:
#         features = json.loads(args.features)
#     elif os.path.exists(meta_path):
#         features = json.load(open(meta_path, "r")).get("features")
#     else:
#         features = input("Enter comma-separated feature names (in order): ").strip().split(",")

#     # Try to initialize LLM parser
#     try:
#         llm_agent = LLMQueryAgent()
#         print("LLM parser available.")
#     except Exception:
#         llm_agent = None
#         print("LLM parser not available (set OPENAI_API_KEY to enable).")

#     run_cli(features, model_path, scaler_path, llm_agent=llm_agent)

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
    print("\n🧠 Loading latest trained model...")
    model, scaler, feature_names, metadata, task_type, dataset_name = load_latest_model()
    print(f"✅ Model loaded ({task_type}) for dataset: {dataset_name}")
    print(f"💡 Features: {len(feature_names)} | {', '.join(feature_names[:8])}...")

    print("\nYou can enter feature=value pairs or natural queries.")
    print("Example:")
    print("  Glucose=120, BMI=30, Age=40")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter values (or natural query): ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting CLI.")
            break
        if not query:
            continue

        try:
            pred, conf = predict_with_llm(query, model, scaler, feature_names)

            if task_type == "classification":
                print(f"🧾 Prediction: Class_{int(pred)} (Confidence={conf})")
            else:
                print(f"🧾 Predicted Value: {pred:.4f}")

        except Exception as e:
            print("❌ Error:", e)
            print("Try again with feature=value format (e.g., Glucose=120, BMI=30).")

if __name__ == "__main__":
    main()

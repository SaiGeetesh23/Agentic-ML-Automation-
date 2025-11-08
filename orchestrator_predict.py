# # orchestrator_predict.py
# import os
# from agents.data_agent import DataAgent
# from agents.predictor_agent import PredictorAgent
# from agents.llm_agent import LLMQueryAgent

# def run_end_to_end(data_path, target, model_name=None, llm_model="gpt-4o-mini"):
#     # 1. Use existing DataAgent to produce splits and feature list
#     da = DataAgent(out_dir="artifacts")
#     data_out = da.run(data_path=data_path, target=target)
#     split_paths = data_out["split_paths"]
#     features = data_out["features"]
#     task_id = data_out["task_id"]

#     # 2. Train an auto-selected model with PredictorAgent
#     pa = PredictorAgent(out_dir="artifacts/models")
#     train_out = pa.train(split_paths=split_paths, features=features, model_name=model_name)
#     print("Trained model:", train_out["model_path"])
#     print("Task type:", train_out["task_type"])
#     print("Metrics:", train_out["metrics"])

#     # 3. Start LLM agent for parsing user queries (requires OPENAI_API_KEY)
#     try:
#         llm = LLMQueryAgent(model_name=llm_model)
#     except Exception as e:
#         print("LLM agent could not be initialized:", e)
#         llm = None

#     return {
#         "task_id": task_id,
#         "features": features,
#         "train_out": train_out,
#         "llm": llm,
#         "predictor": pa
#     }

# if __name__ == "__main__":
#     # Example: python orchestrator_predict.py
#     # out = run_end_to_end("diabetes.csv", target="Outcome")
#     # print("Ready. Use predict_cli.py to make predictions interactively.")

#     out = run_end_to_end("data.csv", target="diagnosis")
#     print("Ready. Use predict_cli.py to make predictions interactively.")


from agents.data_agent import DataAgent
from agents.predictor_agent import train_and_evaluate

def main():
    dataset_path = "data/data.csv"
    target_col = "diagnosis"
    # dataset_path = "data/diabetes.csv"
    # target_col = "Outcome"
    # dataset_path = "data/Housing.csv"
    # target_col = "price"
    # dataset_path = "data/GOOG.csv"
    # target_col = "adjClose"

    data_agent = DataAgent(dataset_path, target_col)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, le = data_agent.execute()

    # ✅ Get feature names after loading dataset
    import pandas as pd
    df = pd.read_csv(dataset_path)
    cols_to_drop = [target_col]
    if "id" in df.columns:
        cols_to_drop.append("id")

    feature_names = df.drop(columns=cols_to_drop).columns.tolist()

    model, metrics = train_and_evaluate(X_train, X_val, y_train, y_val, scaler, feature_names)

if __name__ == "__main__":
    main()

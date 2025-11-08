
from agents.data_agent import DataAgent
from agents.predictor_agent import train_and_evaluate
import pandas as pd
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
    df = pd.read_csv(dataset_path)
    cols_to_drop = [target_col]
    if "id" in df.columns:
        cols_to_drop.append("id")
    feature_names = df.drop(columns=cols_to_drop).columns.tolist()
    model, metrics = train_and_evaluate(X_train, X_val, y_train, y_val, scaler, feature_names)

if __name__ == "__main__":
    main()

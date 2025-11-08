import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv() 
def generate_experiment_review(metadata_path):
    """
    Uses an LLM to generate a structured review of the latest experiment.
    """
    client = OpenAI()
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    prompt = f"""You are an ML experiment reviewer. Analyze and summarize the following experiment metadata.
    Provide a concise but professional review covering:
            - Dataset name
            - Model type
            - Task type
            - Key performance metrics
            - Strengths
            - Weaknesses
            - Suggested improvements

Metadata:
{json.dumps(metadata, indent=2)}
    """
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        temperature=0.4,
    )
    review = response.output[0].content[0].text.strip()
    os.makedirs("artifacts/reviews", exist_ok=True)
    review_path = "artifacts/reviews/latest_review.txt"
    with open(review_path, "w", encoding="utf-8") as f:
        f.write(review)
    print("\n Experiment Review Generated:\n")
    print(review)
    print(f"\n Review saved to: {review_path}\n")
    return review
def main():
    """
    Automatically find the latest metadata and generate a review.
    """
    latest_metadata = "artifacts/models/latest_metadata.json"
    if not os.path.exists(latest_metadata):
        print(" No experiment metadata found. Run orchestrator_predict.py first.")
        return
    print(" Generating LLM-based review for the latest experiment...")
    generate_experiment_review(latest_metadata)

if __name__ == "__main__":
    main()

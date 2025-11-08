# from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
# from langchain_core.output_parsers import StrOutputParser
# import json
# import os

# # 1. Define the PromptTemplate
# PROMPT = PromptTemplate(
#     input_variables=["metrics", "report", "plots", "task_id"],
#     template="""
# You are a concise experimental reviewer. Given the following experiment outputs, produce:
# 1) A short 4-sentence abstract-style summary.
# 2) Three bullet critiques (concise).
# 3) Three concrete next-step recommendations (one sentence each).

# Input:
# Task ID: {task_id}
# Metrics: {metrics}
# Key plots: {plots}
# Classification report (JSON): {report}

# Output format (JSON):
# {{"abstract":"...", "critiques":["...","...","..."], "recommendations":["...","...","..."]}}
# """
# )

# class ReviewerAgentLangChain:
#     def __init__(self, openai_api_key=None, model_name="gpt-4o-mini"):
#         # Initialize the LLM
#         # Replace "OPENAI_API" with your actual API key or ensure it's set in your environment
#         self.llm = ChatOpenAI(temperature=0.0, openai_api_key=os.environ.get("OPENAI_API_KEY"), model=model_name)
        
#         # 2. Define the runnable sequence (LCEL)
#         # This replaces the need for LLMChain
#         self.runnable_chain = PROMPT | self.llm | StrOutputParser()

#     def run(self, task_id: str, metrics: str, report: str, plots: str) -> dict:
#         input_data = {
#             "task_id": task_id,
#             "metrics": metrics,
#             "report": report,
#             "plots": plots
#         }
        
#         # Invoke the runnable sequence
#         resp_string = self.runnable_chain.invoke(input_data)
        
#         # The LLM is instructed to return a JSON string, so we try to parse it
#         try:
#             # Clean up the response string if necessary (e.g., removing '```json' or trailing '```')
#             if resp_string.startswith("```json"):
#                 resp_string = resp_string.strip().lstrip("```json").rstrip("```")
#             out = json.loads(resp_string)
#         except json.JSONDecodeError:
#             print(f"Warning: Failed to parse JSON. Raw output: {resp_string[:100]}...")
#             out = {"raw": resp_string}
        
#         return out

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

    prompt = f"""
You are an ML experiment reviewer. Analyze and summarize the following experiment metadata.
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

    # Save review
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

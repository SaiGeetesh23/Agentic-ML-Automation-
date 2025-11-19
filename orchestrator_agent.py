import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage, SystemMessage
from agents.tool_wrappers import (
    run_data_agent_tool,
    train_model_tool,
    predict_values_tool,
    run_analysis_tool,
    generate_review_tool
)

def main():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    tools = [
        run_data_agent_tool,
        train_model_tool,
        predict_values_tool,
        run_analysis_tool,
        generate_review_tool
    ]
    llm_with_tools = llm.bind_tools(tools)
    print("\n LangChain ML Orchestrator Ready!")
    print("Examples:")
    print("- load dataset data/diabetes.csv target Outcome")
    print("- train model")
    print("- predict open=101 high=103 low=99")
    print("- run analysis")
    print("- generate review\n")
    while True:
        query = input("\n You: ")

        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = llm_with_tools.invoke([
            SystemMessage(content="""
You are an ML Orchestrator.

RULES:
1. You MUST always use a tool if one is relevant.
2. NEVER answer with normal text if a tool can handle the request.
3. If user asks anything related to: loading dataset, preprocessing, training,
   predicting, analysis, or review → ALWAYS call the correct tool.
4. NEVER ask user follow-up questions if the tool does not require them.
5. Dataset is already prepared if run_data_agent was executed earlier.
                          """),
            HumanMessage(content=query)
        ])
     
        if response.tool_calls:
            for call in response.tool_calls:
                tool_name = call["name"]
                args = call["args"]
                print(f"\n Calling: {tool_name} with args {args}")
                tool = next(t for t in tools if t.name == tool_name)
                output = tool.run(args)
                print("\n Result:")
                print(output)
        else:
            print("\nAgent:", response.content)
if __name__ == "__main__":
    main()

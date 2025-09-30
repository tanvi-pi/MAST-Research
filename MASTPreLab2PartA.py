!pip install langchain langchain-openai langgraph datasets
!pip install timeout-decorator
import random
import re
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import os
os.environ["OPENAI_API_KEY"] = "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"

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

dataset = load_dataset("Hothan/OlympiadBench", "OE_MM_maths_en_COMP", split="train")

random.seed(42)
sampled = random.sample(list(dataset), 50)

print(f"Loaded {len(sampled)} problems for evaluation.")

def solver(problem: str) -> str:
    """Problem Solver Agent"""
    messages = [
        SystemMessage(content="You are a mathematical problem solver. "
                              "Solve the problem step by step and clearly state the final boxed answer."),
        HumanMessage(content=problem),
    ]
    return llm.invoke(messages).content

def verifier(problem: str, solution: str) -> str:
    """Verifier Agent"""
    messages = [
        SystemMessage(content="You are a verifier. Check the provided solution for logical or arithmetic errors. "
                              "Respond with either 'APPROVED' if correct, or 'REVISE:' followed by feedback."),
        HumanMessage(content=f"Problem:\n{problem}\n\nSolution Attempt:\n{solution}")
    ]
    return llm.invoke(messages).content

def reviser(problem: str, solution: str, feedback: str) -> str:
    """Reviser Agent"""
    messages = [
        SystemMessage(content="You are a reviser. Based on verifier feedback, fix the solution and provide a corrected final boxed answer."),
        HumanMessage(content=f"Problem:\n{problem}\n\nSolution Attempt:\n{solution}\n\nVerifier Feedback:\n{feedback}")
    ]
    return llm.invoke(messages).content

def judge(problem: str, solution: str, alt_solution: str = None) -> str:
    """Final Judge Agent"""
    messages = [
        SystemMessage(content="You are the final judge. Compare the given solutions and provide the cleanest, most correct final boxed answer."),
        HumanMessage(content=f"Problem:\n{problem}\n\nSolution 1:\n{solution}\n\nSolution 2:\n{alt_solution}")
    ]
    return llm.invoke(messages).content
from timeout_decorator import timeout, TimeoutError
@timeout(60)  # 60-second timeout per problem
def multi_agent_pipeline(problem: str) -> str:
    attempt = solver(problem)
    for _ in range(2):
        feedback = verifier(problem, attempt)
        if "APPROVED" in feedback.upper():
            return attempt
        elif "REVISE:" in feedback:
            attempt = reviser(problem, attempt, feedback)
    return judge(problem, attempt)
def extract_boxed_answer(output: str):
"""Extract predicted answer from agent output."""
    return output.split("Answer:")[-1].strip() if "Answer:" in output else output.strip()

correct = 0
results = []
counter = 1

for ex in sampled:
    print(counter)
    problem, gold = ex["question"], ex["final_answer"]
    system_answer = multi_agent_pipeline(problem)
    pred = extract_boxed_answer(system_answer)

    results.append((problem, gold, pred, system_answer))
    if pred in gold:   # handles multiple valid answers
        correct += 1
    counter = 1 + counter

print(f"Accuracy: {correct} / {len(sampled)} = {100*correct/len(sampled):.2f}%")
import json

execution_traces = []
counter1 = 10
counter2 = 20

for ex in sampled:
    print(counter2)
    problem, gold = ex["question"], ex["final_answer"]

    trace = {"problem": problem, "gold": gold, "agents": []}

    attempt = solver(problem)
    if hasattr(attempt, "content"):
        attempt = attempt.content
    trace["agents"].append({"role": "solver", "output": attempt})

    for _ in range(2):
        print(counter1)
        feedback = verifier(problem, attempt)
        if hasattr(feedback, "content"):
            feedback = feedback.content
        trace["agents"].append({"role": "verifier", "output": feedback})

        if feedback.strip().upper().startswith("REVISE"):
            attempt = reviser(problem, attempt, feedback)
            if hasattr(attempt, "content"):
                attempt = attempt.content
            trace["agents"].append({"role": "reviser", "output": attempt})
        else:
            break
        counter1 = counter1+1
    counter2 = counter2+1

    final = judge(problem, attempt)
    if hasattr(final, "content"):
        final = final.content
    trace["agents"].append({"role": "judge", "output": final})

    execution_traces.append(trace)

with open("execution_traces.json", "w") as f:
    json.dump(execution_traces, f, indent=2)

print("Saved execution traces to execution_traces.json")

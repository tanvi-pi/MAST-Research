
import random
import re
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
from collections import Counter


import os
os.environ["OPENAI_API_KEY"] = "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"


llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

annotations = []
failure_counts = Counter()
multimodal_flags = []
correct = 0
i = 0


def extract_boxed_answer(output: str):
    """Extract predicted answer from agent output."""
    return output.split("Answer:")[-1].strip() if "Answer:" in output else output.strip()

def clean_json_response(resp: str):
    """Strip code blocks and extra text from LLM JSON output."""
    if resp.startswith("```"):
        resp = "\n".join(resp.splitlines()[1:-1])
    return resp.strip()

with open("execution_traces.json", "r") as f:
    traces = json.load(f)

annotations = []
failure_counts = Counter()
multimodal_flags = []
correct = 0

MAST_PROMPT = """
You are MAST, a multi-agent trace analyzer.

For each agent in the trace, identify failures in reasoning, calculation, or evaluation. 
Examples of failure modes include:
- FM-1.1: Disobey Task Specification – Agent fails to follow task constraints or requirements.
- FM-1.2: Disobey Role Specification – Agent acts outside assigned role or responsibilities.
- FM-1.3: Step Repetition – Agent repeats steps unnecessarily despite completion.
- FM-1.4: Loss of Conversation History – Agent ignores recent context and reverts to an earlier state.
- FM-1.5: Unaware of Termination Conditions – Agent continues interaction past required stopping points.
- FM-2.1: Conversation Reset – Dialogue restarts unexpectedly, losing context.
- FM-2.2: Fail to Ask for Clarification – Agent does not request additional information when needed.
- FM-2.3: Task Derailment – Agent deviates from intended objectives, producing irrelevant outputs.
- FM-2.4: Information Withholding – Agent fails to share critical information with others.
- FM-2.5: Ignored Other Agent's Input – Agent disregards input or recommendations from other agents.
- FM-2.6: Action-Reasoning Mismatch – Agent’s actions contradict its own reasoning or internal conclusions.
- FM-3.1: Premature Termination – Ending task or dialogue before objectives are met or required information is obtained.
- FM-3.2: Weak Verification – Verification is incomplete, superficial, or insufficiently rigorous.
- FM-3.3: No or Incorrect Verification – Agent fails to check or confirm outputs properly, allowing errors to propagate.


Return JSON with:
{
  "annotations": [
    {
      "role": "<agent role>",
      "output": "<agent output>",
      "failure_modes": [
        {
          "code": "<failure code>",
          "justification": "<short explanation>"
        }
      ]
    }
  ]
}
Do not include trivial formatting differences as failures.
"""

for trace in traces:
    print(i)
    problem = trace["problem"]
    gold = trace["gold"] if isinstance(trace["gold"], str) else trace["gold"][0]
    agents = trace["agents"]

    trace_summary = f"""Problem: {problem}
Gold Answer: {gold}

Execution Trace:
"""
    for agent in agents:
        trace_summary += f"Agent ({agent['role']}):\n{agent['output']}\n\n"

    if any(k in problem.lower() for k in ["shape", "triangle", "diagram", "figure", "parallel", "angle", "<image"]):
        multimodal_flags.append({"problem": problem[:80]+"...", "suspected_multimodal": True})


    messages = [
        SystemMessage(content=MAST_PROMPT),
        HumanMessage(content=f"Trace:\n{trace_summary}\nPlease annotate any failure modes.")
    ]

    response = llm.invoke(messages).content
    cleaned = clean_json_response(response)
    try:
        annotation = json.loads(cleaned)
    except json.JSONDecodeError:
        annotation = {"annotations": [], "correct": False}


    norm_pred = re.sub(r"[\$\(\)]", "", str(extract_boxed_answer(agents[-1]["output"]))).strip()
    norm_gold = re.sub(r"[\$\(\)]", "", str(gold)).strip()
    annotation["correct"] = norm_pred == norm_gold
    if annotation["correct"]:
        correct += 1


    for agent_annotation in annotation.get("annotations", []):
        for fm in agent_annotation.get("failure_modes", []):
            failure_counts[fm["code"]] += 1

    annotations.append(annotation)
    i = i+1


total_traces = len(traces)
print("MAST Failure Mode Analysis Summary:")
print(f"Total Traces Analyzed: {total_traces}")
print(f"Correct Answers: {correct} ({100 * correct / total_traces:.2f}%)\n")
print("Failure Mode Frequencies:")
for fm, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
    print(f"{fm}: {count} occurrences ({100 * count / total_traces:.2f}%)")

print("\n5 Annotations Preview:")
print(json.dumps(annotations, indent=2))


with open("mast_annotations.json", "w") as f:
    json.dump(annotations, f, indent=2)
print("\n Saved MAST annotations to mast_annotations.json")



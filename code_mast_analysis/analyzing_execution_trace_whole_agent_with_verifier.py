# Cell 2: Imports
import random
import re
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
from collections import Counter


import os
os.environ["OPENAI_API_KEY"] = "API_KEY"
# Cell 4: Initialize LLM


llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

annotations = []
failure_counts = Counter()
multimodal_flags = []
correct = 0
i = 0


# ---- Helper functions ----
def extract_boxed_answer(output: str):
    """Extract predicted answer from agent output."""
    return output.split("Answer:")[-1].strip() if "Answer:" in output else output.strip()

def clean_json_response(resp: str):
    """Strip code blocks and extra text from LLM JSON output."""
    if resp.startswith("```"):
        resp = "\n".join(resp.splitlines()[1:-1])
    return resp.strip()

# ---- Load traces ----
with open("MAST-Research/json_execution_traces/execution_traces_with_reasoning_verifier.json", "r") as f:
    traces = json.load(f)

# ---- Initialize counters ----
annotations = []
failure_counts = Counter()
multimodal_flags = []
correct = 0

# ---- Define MAST prompt with failure modes ----
MAST_PROMPT = """
You are MAST, a multi-agent trace analyzer.

Your task is to:
1. **Define the problem type and what's required to solve it**
2. **Identify specific issues in the multi-agent conversation**
3. **Map those issues to failure modes**

**CRITICAL**: The gold answer is provided. Use it to:
- Determine if the solution is correct or incorrect
- If INCORRECT: Identify which agent(s) introduced the error and why
- If CORRECT: Check if there were any concerning patterns that could cause failures in other cases

Analyze the ENTIRE multi-agent conversation as a cohesive system.


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


**IMPORTANT INSTRUCTIONS**:
- First, determine if the final answer matches the gold answer
- If INCORRECT: Focus on identifying the root cause - which agent made the critical error?
- If CORRECT: Only flag failures if there were substantive issues in the reasoning process
- Don't flag minor inefficiencies or harmless repetition for correct solutions

Return JSON with:
{
  "problem_analysis": {
    "problem_type": "<e.g., 'Quadratic Equation', 'Geometry - Triangle Properties'>",
    "key_requirements": [
      "<What steps/knowledge are needed to solve this?>"
    ],
    "difficulty_factors": [
      "<What makes this problem challenging?>"
    ]
  },
  
  "solution_correctness": {
    "final_answer": "<Extract the final answer from the trace>",
    "gold_answer": "<The provided gold answer>",
    "is_correct": "<true or false>",
    "explanation": "<Why is it correct/incorrect?>"
  },
  
  "identified_issues": [
    {
      "issue_description": "<Specific problem you observed>",
      "where_it_occurred": "<Which agent(s) and at what step>",
      "impact_on_solution": "<How this affected the final answer - be specific>",
      "severity": "<CRITICAL (caused wrong answer) | MODERATE (risky pattern) | MINOR (inefficiency)>",
      "applicable_failure_modes": [
        {
          "code": "<FM-X.X>",
          "why_it_applies": "<Explain how the failure mode matches this issue>"
        }
      ]
    }
  ],
  
  "trace_level_analysis": "<Overall assessment - emphasize whether agents caught or missed errors>",
  
  "conversation_flow_assessment": {
    "coordination_quality": "<How well did agents build on each other's work?>",
    "error_propagation": "<Did errors get caught or did they compound?>"
  }
}

Do not include trivial formatting differences as failures.
"""

# ---- Loop over traces ----
for trace in traces[:50]:
    print(i)
    problem = trace["problem"]
    gold = trace["gold"] if isinstance(trace["gold"], str) else trace["gold"][0]
    agents = trace["agents"]

    # Build trace summary
    trace_summary = f"""Problem: {problem}
Gold Answer: {gold}

Execution Trace:
"""
    for idx, agent in enumerate(agents, 1):
        trace_summary += f"\n[Step {idx}] {agent['role'].upper()}:\n{agent['output']}\n"
        trace_summary += "-" * 80 + "\n"
    
    trace_summary += f"\n=== END OF CONVERSATION ===\n"
    trace_summary += f"Total steps: {len(agents)}\n"

    # Flag suspected multimodal problems
    if any(k in problem.lower() for k in ["shape", "triangle", "diagram", "figure", "parallel", "angle", "<image"]):
        multimodal_flags.append({"problem": problem[:80]+"...", "suspected_multimodal": True})

    # ---- Call MAST ----
    messages = [
        SystemMessage(content=MAST_PROMPT),
        HumanMessage(content=f"{trace_summary}\n\nAnalyze this ENTIRE conversation trace. Focus on how agents interacted and whether the conversation was coherent and effective.")
    ]

    response = llm.invoke(messages).content
    cleaned = clean_json_response(response)
    try:
        annotation = json.loads(cleaned)
    except json.JSONDecodeError:
        annotation = {
            "trace_level_analysis": "",
            "conversation_flow_issues": [],
            "agent_annotations": [],
            "correct": False
        }

    # Check correctness
    norm_pred = re.sub(r"[\$\(\)]", "", str(extract_boxed_answer(agents[-1]["output"]))).strip()
    norm_gold = re.sub(r"[\$\(\)]", "", str(gold)).strip()
    annotation["correct"] = norm_pred == norm_gold
    if annotation["correct"]:
        correct += 1

    # Collect failure modes
    for agent_annotation in annotation.get("agent_annotations", []):
        for fm in agent_annotation.get("failure_modes", []):
            failure_counts[fm["code"]] += 1
    
    for flow_issue in annotation.get("conversation_flow_issues", []):
        failure_counts["FLOW-ISSUE"] += 1

    annotations.append(annotation)
    i = i+1

# ---- Print summary ----
total_traces = len(traces)
print("MAST Failure Mode Analysis Summary:")
print(f"Total Traces Analyzed: {total_traces}")
print(f"Correct Answers: {correct} ({100 * correct / total_traces:.2f}%)\n")
print("Failure Mode Frequencies:")
for fm, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
    print(f"{fm}: {count} occurrences ({100 * count / total_traces:.2f}%)")

# ---- Save results ----
with open("mast_holistic_annotations.json", "w") as f:
    json.dump(annotations, f, indent=2)
print("\n Saved MAST holistic annotations to mast_holistic_annotations.json")

total_flow_issues = sum(len(ann.get("conversation_flow_issues", [])) for ann in annotations)
print(f"\n Conversation Flow Issues Detected: {total_flow_issues}")
print(f"   Average per trace: {total_flow_issues / total_traces:.2f}")

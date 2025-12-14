"""
Multi-Plan Selection - Tree of Thoughts Approach
Generates multiple plans and selects the best one using evaluation.
"""

from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import os
from dotenv import load_dotenv
load_dotenv()


from langchain_openai import ChatOpenAI

class PlanCandidate(TypedDict):
    plan: str
    score: float
    reasoning: str

class MultiPlanState(TypedDict):
    task: str
    messages: Annotated[List, operator.add]
    candidates: List[PlanCandidate]
    best_plan: PlanCandidate
    num_candidates: int

def generate_multiple_plans(state: MultiPlanState) -> MultiPlanState:
    """Generate multiple candidate plans using different sampling."""
    llm = ChatOpenAI(model="gpt-4", temperature=0.8)
    
    system_msg = SystemMessage(content="""You are a creative planner. 
Generate a detailed plan to accomplish the given task. 
Be specific and actionable.""")
    
    task_msg = HumanMessage(content=f"""Task: {state['task']}

Generate a step-by-step plan to accomplish this task. 
Provide a unique approach.""")
    
    candidates = []
    
    # Generate multiple candidates with temperature sampling
    for i in range(state["num_candidates"]):
        response = llm.invoke([system_msg, task_msg])
        candidates.append({
            "plan": response.content,
            "score": 0.0,
            "reasoning": ""
        })
        print(f"Generated plan {i+1}/{state['num_candidates']}")
    
    return {
        **state,
        "candidates": candidates,
        "messages": state["messages"] + [system_msg, task_msg]
    }

def evaluate_plans(state: MultiPlanState) -> MultiPlanState:
    """Evaluate each candidate plan and assign scores."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(content="""You are an expert evaluator.
Evaluate the given plan based on:
1. Feasibility (can it be executed?)
2. Completeness (does it cover all aspects?)
3. Efficiency (is it optimized?)
4. Clarity (is it easy to understand?)

Provide a score from 0-10 and brief reasoning.""")
    
    evaluated_candidates = []
    
    for i, candidate in enumerate(state["candidates"]):
        eval_prompt = f"""Task: {state['task']}

Plan to evaluate:
{candidate['plan']}

Provide your evaluation in this format:
Score: [0-10]
Reasoning: [your explanation]"""
        
        response = llm.invoke([system_msg, HumanMessage(content=eval_prompt)])
        response_text = response.content
        
        # Parse score and reasoning
        score = 5.0  # default
        reasoning = response_text
        
        for line in response_text.split('\n'):
            if line.startswith("Score:"):
                try:
                    score = float(line.replace("Score:", "").strip())
                except:
                    pass
            elif line.startswith("Reasoning:"):
                reasoning = line.replace("Reasoning:", "").strip()
        
        evaluated_candidates.append({
            **candidate,
            "score": score,
            "reasoning": reasoning
        })
        
        print(f"Evaluated plan {i+1}: Score = {score}")
    
    return {
        **state,
        "candidates": evaluated_candidates
    }

def select_best_plan(state: MultiPlanState) -> MultiPlanState:
    """Select the plan with the highest score."""
    best = max(state["candidates"], key=lambda x: x["score"])
    
    return {
        **state,
        "best_plan": best
    }

def create_multi_plan_graph():
    """Build the multi-plan selection workflow."""
    workflow = StateGraph(MultiPlanState)
    
    # Add nodes
    workflow.add_node("generate", generate_multiple_plans)
    workflow.add_node("evaluate", evaluate_plans)
    workflow.add_node("select", select_best_plan)
    
    # Add edges
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "evaluate")
    workflow.add_edge("evaluate", "select")
    workflow.add_edge("select", END)
    
    return workflow.compile()

# Example usage
if __name__ == "__main__":
    graph = create_multi_plan_graph()
    
    initial_state = {
        "task": "Organize a team building event for 20 people with a $2000 budget",
        "messages": [],
        "candidates": [],
        "best_plan": {},
        "num_candidates": 3
    }
    
    print("Starting Multi-Plan Selection Agent...")
    print(f"Task: {initial_state['task']}")
    print(f"Generating {initial_state['num_candidates']} candidate plans...\n")
    
    result = graph.invoke(initial_state)
    
    print("\n=== All Candidate Plans ===")
    for i, candidate in enumerate(result["candidates"]):
        print(f"\n--- Plan {i+1} (Score: {candidate['score']}) ---")
        print(candidate['plan'][:300] + "..." if len(candidate['plan']) > 300 else candidate['plan'])
        print(f"Evaluation: {candidate['reasoning']}")
    
    print("\n\n=== SELECTED BEST PLAN ===")
    print(f"Score: {result['best_plan']['score']}")
    print(f"\n{result['best_plan']['plan']}")
    print(f"\nReasoning: {result['best_plan']['reasoning']}")
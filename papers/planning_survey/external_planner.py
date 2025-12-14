"""
External Planner-Aided Planning
Uses LLM to formalize tasks and external symbolic planner for execution.
"""

from typing import TypedDict, List, Annotated, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import os
import json

os.environ["OPENAI_API_KEY"] = "your-api-key-here"

from langchain_openai import ChatOpenAI

class ExternalPlannerState(TypedDict):
    task: str
    messages: Annotated[List, operator.add]
    formalized_task: Dict
    plan: List[str]
    execution_result: str

def formalize_task(state: ExternalPlannerState) -> ExternalPlannerState:
    """Use LLM to formalize the task into structured format."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(content="""You are an expert at formalizing tasks.
Convert natural language tasks into structured JSON format with:
- goal: the main objective
- initial_state: current conditions
- constraints: limitations or rules
- actions: available actions to take
- objects: entities involved in the task

Be precise and complete.""")
    
    task_msg = HumanMessage(content=f"""Task: {state['task']}

Formalize this task into a structured JSON format.
Include goal, initial_state, constraints, available actions, and objects.""")
    
    response = llm.invoke([system_msg, task_msg])
    
    # Parse JSON response
    try:
        # Extract JSON from response
        content = response.content
        start = content.find('{')
        end = content.rfind('}') + 1
        json_str = content[start:end]
        formalized = json.loads(json_str)
    except:
        # Fallback structure
        formalized = {
            "goal": state["task"],
            "initial_state": {},
            "constraints": [],
            "actions": [],
            "objects": []
        }
    
    return {
        **state,
        "formalized_task": formalized,
        "messages": state["messages"] + [system_msg, task_msg, response]
    }

def symbolic_planning(state: ExternalPlannerState) -> ExternalPlannerState:
    """Use symbolic planner to generate optimal plan."""
    # Simulate a symbolic planner (in practice, this would be PDDL solver, etc.)
    formalized = state["formalized_task"]
    
    # Simple planning logic based on formalized structure
    goal = formalized.get("goal", "")
    actions = formalized.get("actions", [])
    constraints = formalized.get("constraints", [])
    
    plan = []
    
    # Generate plan based on available actions
    if actions:
        # Simple sequential planning
        for action in actions:
            if isinstance(action, dict):
                action_name = action.get("name", str(action))
            else:
                action_name = str(action)
            plan.append(f"Execute: {action_name}")
    else:
        # Generic planning steps
        plan = [
            "1. Analyze requirements from formalized task",
            "2. Identify necessary resources and dependencies",
            "3. Sequence actions according to constraints",
            "4. Execute actions in optimal order",
            "5. Verify goal achievement"
        ]
    
    # Add constraint checks
    if constraints:
        plan.insert(0, f"Verify constraints: {', '.join([str(c) for c in constraints])}")
    
    plan.append(f"Achieve goal: {goal}")
    
    return {
        **state,
        "plan": plan
    }

def execute_plan(state: ExternalPlannerState) -> ExternalPlannerState:
    """Simulate plan execution."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    execution_prompt = f"""Given this formalized task and generated plan, 
provide a summary of the execution:

Formalized Task:
{json.dumps(state['formalized_task'], indent=2)}

Generated Plan:
{chr(10).join(state['plan'])}

Describe how this plan would be executed step-by-step."""
    
    response = llm.invoke([HumanMessage(content=execution_prompt)])
    
    return {
        **state,
        "execution_result": response.content
    }

def create_external_planner_graph():
    """Build the external planner workflow."""
    workflow = StateGraph(ExternalPlannerState)
    
    # Add nodes
    workflow.add_node("formalize", formalize_task)
    workflow.add_node("plan", symbolic_planning)
    workflow.add_node("execute", execute_plan)
    
    # Add edges
    workflow.set_entry_point("formalize")
    workflow.add_edge("formalize", "plan")
    workflow.add_edge("plan", "execute")
    workflow.add_edge("execute", END)
    
    return workflow.compile()

# Example usage
if __name__ == "__main__":
    graph = create_external_planner_graph()
    
    initial_state = {
        "task": "Move all boxes from room A to room B, where boxes must be stacked by size (largest at bottom) and fragile boxes must be on top",
        "messages": [],
        "formalized_task": {},
        "plan": [],
        "execution_result": ""
    }
    
    print("Starting External Planner-Aided Agent...")
    print(f"Task: {initial_state['task']}\n")
    
    result = graph.invoke(initial_state)
    
    print("=== FORMALIZED TASK ===")
    print(json.dumps(result["formalized_task"], indent=2))
    
    print("\n\n=== GENERATED PLAN ===")
    for step in result["plan"]:
        print(f"  {step}")
    
    print("\n\n=== EXECUTION SUMMARY ===")
    print(result["execution_result"])
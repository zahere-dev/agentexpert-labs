"""
Reflection and Refinement - Reflexion Approach
Iteratively improves plans through reflection on failures and feedback.
"""

from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import operator
import os

os.environ["OPENAI_API_KEY"] = "your-api-key-here"

from langchain_openai import ChatOpenAI

class ReflectionState(TypedDict):
    task: str
    messages: Annotated[List, operator.add]
    current_plan: str
    execution_feedback: str
    reflection: str
    iteration: int
    max_iterations: int
    success: bool
    plan_history: Annotated[List, operator.add]

def generate_initial_plan(state: ReflectionState) -> ReflectionState:
    """Generate the initial plan for the task."""
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    system_msg = SystemMessage(content="""You are a planning assistant.
Generate a detailed, step-by-step plan to accomplish the given task.""")
    
    task_msg = HumanMessage(content=f"""Task: {state['task']}

Create a comprehensive plan with specific, actionable steps.""")
    
    response = llm.invoke([system_msg, task_msg])
    
    return {
        **state,
        "current_plan": response.content,
        "iteration": 1,
        "messages": [system_msg, task_msg, response],
        "plan_history": [{"iteration": 1, "plan": response.content}]
    }

def simulate_execution(state: ReflectionState) -> ReflectionState:
    """Simulate execution and generate realistic feedback."""
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    feedback_prompt = f"""Simulate executing this plan and identify potential issues:

Plan:
{state['current_plan']}

Provide realistic feedback about:
1. What worked well
2. What failed or could fail
3. Missing steps or considerations
4. Resource or constraint violations

Be specific and constructive."""
    
    response = llm.invoke([HumanMessage(content=feedback_prompt)])
    
    # Simple success evaluation based on feedback
    feedback_lower = response.content.lower()
    success = (
        "success" in feedback_lower or 
        "no major issues" in feedback_lower or
        state["iteration"] >= state["max_iterations"]
    )
    
    return {
        **state,
        "execution_feedback": response.content,
        "success": success
    }

def reflect_on_feedback(state: ReflectionState) -> ReflectionState:
    """Reflect on the execution feedback to identify improvements."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(content="""You are a reflective agent that learns from failures.
Analyze execution feedback and generate insights for improvement.""")
    
    reflection_prompt = f"""Plan (Iteration {state['iteration']}):
{state['current_plan']}

Execution Feedback:
{state['execution_feedback']}

Reflect deeply on:
1. Root causes of any failures
2. What assumptions were wrong
3. What was overlooked
4. Key insights for the next iteration

Provide actionable reflections."""
    
    response = llm.invoke([system_msg, HumanMessage(content=reflection_prompt)])
    
    return {
        **state,
        "reflection": response.content,
        "messages": state["messages"] + [AIMessage(content=f"Reflection: {response.content}")]
    }

def refine_plan(state: ReflectionState) -> ReflectionState:
    """Refine the plan based on reflection."""
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    system_msg = SystemMessage(content="""You are an expert planner.
Use the reflection to create an improved plan that addresses previous issues.""")
    
    refine_prompt = f"""Original Task: {state['task']}

Previous Plan:
{state['current_plan']}

Reflection on Failures:
{state['reflection']}

Create an IMPROVED plan that:
- Addresses the identified issues
- Incorporates the insights from reflection
- Is more robust and complete

Provide a refined step-by-step plan."""
    
    response = llm.invoke([system_msg, HumanMessage(content=refine_prompt)])
    
    return {
        **state,
        "current_plan": response.content,
        "iteration": state["iteration"] + 1,
        "messages": state["messages"] + [response],
        "plan_history": state["plan_history"] + [
            {"iteration": state["iteration"] + 1, "plan": response.content}
        ]
    }

def should_continue(state: ReflectionState) -> str:
    """Decide whether to continue refining or end."""
    if state["success"] or state["iteration"] >= state["max_iterations"]:
        return "end"
    return "continue"

def create_reflection_graph():
    """Build the reflection and refinement workflow."""
    workflow = StateGraph(ReflectionState)
    
    # Add nodes
    workflow.add_node("generate", generate_initial_plan)
    workflow.add_node("execute", simulate_execution)
    workflow.add_node("reflect", reflect_on_feedback)
    workflow.add_node("refine", refine_plan)
    
    # Add edges
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "execute")
    workflow.add_edge("execute", "reflect")
    
    # Conditional edge after reflection
    workflow.add_conditional_edges(
        "reflect",
        should_continue,
        {
            "continue": "refine",
            "end": END
        }
    )
    
    # Loop back to execute after refinement
    workflow.add_edge("refine", "execute")
    
    return workflow.compile()

# Example usage
if __name__ == "__main__":
    graph = create_reflection_graph()
    
    initial_state = {
        "task": "Create a marketing campaign for a new eco-friendly water bottle targeting millennials",
        "messages": [],
        "current_plan": "",
        "execution_feedback": "",
        "reflection": "",
        "iteration": 0,
        "max_iterations": 3,
        "success": False,
        "plan_history": []
    }
    
    print("Starting Reflection and Refinement Agent...")
    print(f"Task: {initial_state['task']}")
    print(f"Max Iterations: {initial_state['max_iterations']}\n")
    
    result = graph.invoke(initial_state)
    
    print("=== PLAN EVOLUTION ===\n")
    for plan_version in result["plan_history"]:
        print(f"--- Iteration {plan_version['iteration']} ---")
        print(plan_version['plan'][:400] + "..." if len(plan_version['plan']) > 400 else plan_version['plan'])
        print()
    
    print(f"\n=== FINAL STATUS ===")
    print(f"Total Iterations: {result['iteration']}")
    print(f"Success: {result['success']}")
    print(f"\nFinal Plan:\n{result['current_plan']}")
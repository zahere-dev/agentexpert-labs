"""
Task Decomposition - Interleaved Approach (ReAct-style)
Dynamically decomposes tasks and plans step-by-step with reasoning.
"""

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import operator
import os
from dotenv import load_dotenv
load_dotenv()

# Set your API key
print("Using OpenAI API Key:", os.getenv("OPENAI_API_KEY"))

from langchain_openai import ChatOpenAI

# class AgentState(TypedDict):
#     messages: Annotated[List, operator.add]
#     task: str
#     current_step: int
#     max_steps: int
#     thought: str
#     action: str
#     observation: str
#     completed: bool

# def initialize_state(state: AgentState) -> AgentState:
#     """Initialize the agent state with the task."""
#     return {
#         **state,
#         "current_step": 0,
#         "max_steps": 10,
#         "completed": False,
#         "messages": [
#             SystemMessage(content="""You are a helpful assistant that breaks down complex tasks.
# For each step:
# 1. Think about what needs to be done (Thought)
# 2. Decide on an action (Action)
# 3. Observe the result (Observation)

# Available actions: search, calculate, write, finish""")
#         ]
#     }

# def think_and_act(state: AgentState) -> AgentState:
#     """Generate thought and action for current step."""
#     llm = ChatOpenAI(model="gpt-4", temperature=0)
    
#     # Build prompt with conversation history
#     prompt = f"""Task: {state['task']}
# Current Step: {state['current_step'] + 1}/{state['max_steps']}

# Based on the conversation so far, what is your next thought and action?

# Format your response as:
# Thought: [your reasoning]
# Action: [one of: search, calculate, write, finish]
# Action Input: [what you want to do]"""

#     messages = state["messages"] + [HumanMessage(content=prompt)]
#     response = llm.invoke(messages)
#     response_text = response.content
    
#     # Parse response
#     lines = response_text.strip().split('\n')
#     thought = ""
#     action = ""
#     action_input = ""
    
#     for line in lines:
#         if line.startswith("Thought:"):
#             thought = line.replace("Thought:", "").strip()
#         elif line.startswith("Action:"):
#             action = line.replace("Action:", "").strip()
#         elif line.startswith("Action Input:"):
#             action_input = line.replace("Action Input:", "").strip()
    
#     return {
#         **state,
#         "thought": thought,
#         "action": action,
#         "messages": state["messages"] + [HumanMessage(content=prompt), response],
#         "current_step": state["current_step"] + 1
#     }

# def execute_action(state: AgentState) -> AgentState:
#     """Simulate action execution and generate observation."""
#     action = state["action"]
    
#     # Simulate different actions
#     if action == "finish":
#         observation = "Task completed successfully!"
#         completed = True
#     elif action == "search":
#         observation = f"Search results retrieved for: {state['thought']}"
#         completed = False
#     elif action == "calculate":
#         observation = f"Calculation performed based on: {state['thought']}"
#         completed = False
#     elif action == "write":
#         observation = f"Content written: {state['thought']}"
#         completed = False
#     else:
#         observation = f"Action '{action}' executed"
#         completed = False
    
#     return {
#         **state,
#         "observation": observation,
#         "completed": completed,
#         "messages": state["messages"] + [
#             AIMessage(content=f"Observation: {observation}")
#         ]
#     }

# def should_continue(state: AgentState) -> str:
#     """Decide whether to continue or end."""
#     if state["completed"] or state["current_step"] >= state["max_steps"]:
#         return "end"
#     return "continue"

# # Build the graph
# def create_task_decomposition_graph():
#     workflow = StateGraph(AgentState)
    
#     # Add nodes
#     workflow.add_node("initialize", initialize_state)
#     workflow.add_node("think_act", think_and_act)
#     workflow.add_node("execute", execute_action)
    
#     # Add edges
#     workflow.set_entry_point("initialize")
#     workflow.add_edge("initialize", "think_act")
#     workflow.add_edge("think_act", "execute")
    
#     # Conditional edge
#     workflow.add_conditional_edges(
#         "execute",
#         should_continue,
#         {
#             "continue": "think_act",
#             "end": END
#         }
#     )
    
#     return workflow.compile()

# # Example usage
# if __name__ == "__main__":
#     graph = create_task_decomposition_graph()
    
#     initial_state = {
#         "task": "Plan a birthday party for a 10-year-old",
#         "messages": [],
#         "current_step": 0,
#         "max_steps": 5,
#         "thought": "",
#         "action": "",
#         "observation": "",
#         "completed": False
#     }
    
#     print("Starting Task Decomposition Agent...")
#     print(f"Task: {initial_state['task']}\n")
    
#     result = graph.invoke(initial_state)
    
#     print("\n=== Execution Trace ===")
#     for i, msg in enumerate(result["messages"]):
#         if isinstance(msg, SystemMessage):
#             continue
#         print(f"\n{msg.__class__.__name__}:")
#         print(msg.content[:200] + "..." if len(msg.content) > 200 else msg.content)
    
#     print(f"\n\nCompleted: {result['completed']}")
#     print(f"Total Steps: {result['current_step']}")
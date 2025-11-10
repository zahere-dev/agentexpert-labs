# langgraph_todos_fixed.py
from dotenv import load_dotenv
import requests
load_dotenv()

"""
Simple Write-Todos Agent Example using LangGraph
This demonstrates task decomposition and execution pattern for teaching purposes.

Requirements:
pip install langgraph langchain-openai tavily-python langchain-core
"""

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import os

# ============================================================================
# STEP 1: Define the State
# ============================================================================
class AgentState(TypedDict):
    """State that flows through our graph"""
    user_query: str
    todos: List[str]
    current_todo_index: int
    results: List[str]
    final_answer: str


# ============================================================================
# STEP 2: Initialize Tools
# ============================================================================
# Set your API keys

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def tavily_search(query: str, max_results: int = 3) -> dict:
    """
    Custom method to call Tavily API directly using requests
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary containing search results
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set")
    
    url = "https://api.tavily.com/search"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "query": query,
        "max_results": max_results
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, verify=False)
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Tavily API: {e}")
        return {"results": [], "error": str(e)}


# ============================================================================
# STEP 3: Define Node Functions
# ============================================================================

def write_todos(state: AgentState) -> AgentState:
    """
    Node 1: Decompose the user query into actionable todos
    This is the key 'write_todos' concept!
    """
    print("\n🔍 STEP 1: Writing Todos...")
    
    prompt = f"""
    Given this user query: "{state['user_query']}"
    
    Break it down into 2-4 simple, actionable search tasks.
    Each task should be a specific web search query.
    
    Return ONLY a numbered list, one task per line.
    Example:
    1. Search for X
    2. Search for Y
    3. Search for Z
    """
    
    response = llm.invoke([SystemMessage(content=prompt)])
    
    # Parse the todos from the response
    todos = []
    for line in response.content.strip().split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            # Remove numbering
            todo = line.split('.', 1)[-1].strip()
            if todo:
                todos.append(todo)
    
    print(f"📝 Created {len(todos)} todos:")
    for i, todo in enumerate(todos, 1):
        print(f"   {i}. {todo}")
    
    state['todos'] = todos
    state['current_todo_index'] = 0
    state['results'] = []
    
    return state


def execute_todo(state: AgentState) -> AgentState:
    """
    Node 2: Execute the current todo using Tavily search
    """
    current_idx = state['current_todo_index']
    current_todo = state['todos'][current_idx]
    
    print(f"\n⚙️  STEP 2: Executing Todo {current_idx + 1}/{len(state['todos'])}")
    print(f"   Task: {current_todo}")
    
    # Extract search query from the todo
    search_query = current_todo.replace("Search for", "").strip()
    
    # Execute search using Tavily
    try:
        search_results = tavily_search(query=search_query)
        
        # Extract key information
        summary = f"Results for '{search_query}':\n"
        for result in search_results.get('results', [])[:2]:
            summary += f"- {result.get('content', '')[:200]}...\n"
        
        print(f"   ✅ Completed: Found {len(search_results.get('results', []))} results")
        
    except Exception as e:
        summary = f"Error searching for '{search_query}': {str(e)}"
        print(f"   ❌ Error: {str(e)}")
    
    state['results'].append(summary)
    state['current_todo_index'] += 1
    
    return state


def should_continue(state: AgentState) -> str:
    """
    Router: Decide if we should execute more todos or synthesize
    """
    if state['current_todo_index'] < len(state['todos']):
        return "execute"
    else:
        return "synthesize"


def synthesize_answer(state: AgentState) -> AgentState:
    """
    Node 3: Combine all results into a final answer
    """
    print("\n🎯 STEP 3: Synthesizing Final Answer...")
    
    all_results = "\n\n".join(state['results'])
    
    prompt = f"""
    Original question: {state['user_query']}
    
    Here are the research results:
    {all_results}
    
    Provide a clear, concise answer to the original question based on these results.
    """
    
    response = llm.invoke([SystemMessage(content=prompt)])
    state['final_answer'] = response.content
    
    print("✨ Final answer generated!")
    
    return state


# ============================================================================
# STEP 4: Build the Graph
# ============================================================================

def create_agent():
    """Create and compile the LangGraph agent"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("write_todos", write_todos)
    workflow.add_node("execute_todo", execute_todo)
    workflow.add_node("synthesize", synthesize_answer)
    
    # Define the flow
    workflow.set_entry_point("write_todos")
    
    workflow.add_conditional_edges(
        "execute_todo",
        should_continue,
        {
            "execute": "execute_todo",  # Loop back to execute next todo
            "synthesize": "synthesize"   # Move to synthesis
        }
    )
    
    workflow.add_edge("write_todos", "execute_todo")
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()


# ============================================================================
# STEP 5: Run the Agent
# ============================================================================

def main():
    """Run a simple example"""
    
    agent = create_agent()
    
    # Example query
    user_query = "To skills to learn in 2026?"
    
    print("="*70)
    print("🤖 WRITE-TODOS AGENT DEMO")
    print("="*70)
    print(f"\n💬 User Query: {user_query}\n")
    
    # Run the agent
    initial_state = {
        "user_query": user_query,
        "todos": [],
        "current_todo_index": 0,
        "results": [],
        "final_answer": ""
    }
    
    result = agent.invoke(initial_state)
    
    print("\n" + "="*70)
    print("📊 FINAL RESULT")
    print("="*70)
    print(f"\n{result['final_answer']}\n")


if __name__ == "__main__":
    main()
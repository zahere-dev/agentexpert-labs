"""
Customer Support Agent with File-Based Todo Management
Demonstrates the Claude-style pattern of writing todos to files and reading them back.

Requirements:
pip install langgraph langchain-openai langchain-core
"""

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
import os
import json
from datetime import datetime, time

from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# STEP 1: Define the State
# ============================================================================
class SupportAgentState(TypedDict):
    """State that flows through our support agent"""
    ticket_id: str
    customer_query: str
    todos_file: str
    todos: List[dict]  # Each todo: {"id": int, "task": str, "status": "pending/completed", "result": str}
    current_todo_index: int
    knowledge_base: dict  # Simulated KB
    final_resolution: str


# ============================================================================
# STEP 2: File Management Functions (Claude-style)
# ============================================================================

def write_todos_to_file(todos: List[dict], filename: str) -> None:
    """
    Write todos to a JSON file (like Claude writes to artifacts)
    """
    with open(filename, 'w') as f:
        json.dump({
            "created_at": datetime.now().isoformat(),
            "todos": todos
        }, f, indent=2)
    print(f"📝 Wrote {len(todos)} todos to {filename}")


def read_todos_from_file(filename: str) -> List[dict]:
    """
    Read todos from a JSON file (like Claude reads from artifacts)
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            print(f"📖 Read {len(data['todos'])} todos from {filename}")
            return data['todos']
    except FileNotFoundError:
        print(f"⚠️  File {filename} not found, returning empty list")
        return []


def update_todo_in_file(filename: str, todo_id: int, status: str, result: str = "") -> None:
    """
    Update a specific todo in the file (like Claude updates artifacts)
    """
    data = json.load(open(filename, 'r'))
    for todo in data['todos']:
        if todo['id'] == todo_id:
            todo['status'] = status
            todo['result'] = result
            break
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✏️  Updated todo #{todo_id} in {filename}")


# ============================================================================
# STEP 3: Initialize Tools
# ============================================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def search_knowledge_base(query: str, kb: dict) -> str:
    """
    Simulated knowledge base search (in real world, this could be a vector DB)
    """
    query_lower = query.lower()
    
    # Simple keyword matching
    for topic, info in kb.items():
        if topic.lower() in query_lower:
            return f"KB Article: {info}"
    
    return "No relevant article found in knowledge base."


def search_ticket_database(query: str, max_results: int = 2) -> dict:
    """
    Mock database search for similar tickets and issues
    Simulates searching past customer support tickets
    """
    # Stubbed database of past tickets
    mock_tickets = {
        "shipping": [
            {
                "ticket_id": "TICKET-2024-089",
                "title": "Order delayed - carrier issue",
                "content": "Customer reported order stuck in transit for 8 days. Root cause: Holiday shipping backlog. Resolution: Expedited replacement sent, original order refunded.",
                "resolution_time": "2 days",
                "category": "shipping"
            },
            {
                "ticket_id": "TICKET-2024-112",
                "title": "Package not delivered after 10 days",
                "content": "Order placed 10 days ago, tracking shows 'in transit' with no updates. Customer requested refund. Resolution: Refund processed, new order sent with express shipping at no cost.",
                "resolution_time": "1 day",
                "category": "shipping"
            }
        ],
        "refund": [
            {
                "ticket_id": "TICKET-2024-045",
                "title": "Refund request for damaged item",
                "content": "Customer received damaged product, requested full refund. Resolution: Immediate refund issued, customer kept the item as gesture of goodwill.",
                "resolution_time": "1 day",
                "category": "refund"
            },
            {
                "ticket_id": "TICKET-2024-078",
                "title": "Refund processing time question",
                "content": "Customer asked about standard refund timeline. Informed 5-7 business days, can take up to 10 days depending on bank. Customer satisfied.",
                "resolution_time": "< 1 hour",
                "category": "refund"
            }
        ],
        "payment": [
            {
                "ticket_id": "TICKET-2024-156",
                "title": "Payment failed but amount charged",
                "content": "Authorization hold placed but order not completed. Hold released within 3-5 business days. Advised customer to retry payment.",
                "resolution_time": "30 minutes",
                "category": "payment"
            }
        ],
        "account": [
            {
                "ticket_id": "TICKET-2024-203",
                "title": "Cannot login to account",
                "content": "Password reset link sent. Customer regained access within 10 minutes.",
                "resolution_time": "15 minutes",
                "category": "account"
            }
        ]
    }
    
    query_lower = query.lower()
    results = []
    
    # Search through mock database
    for category, tickets in mock_tickets.items():
        if category in query_lower or any(word in query_lower for word in ["delay", "shipping", "delivery", "order", "refund", "payment", "account"]):
            for ticket in tickets:
                if any(term in ticket['title'].lower() or term in ticket['content'].lower() 
                       for term in query_lower.split()):
                    results.append(ticket)
                    if len(results) >= max_results:
                        break
        if len(results) >= max_results:
            break
    
    # If no specific matches, return some default results
    if not results:
        results = mock_tickets["shipping"][:max_results]
    
    return {
        "results": results[:max_results],
        "total_found": len(results)
    }


# ============================================================================
# STEP 4: Node Functions
# ============================================================================

def plan_support_actions(state: SupportAgentState) -> SupportAgentState:
    """
    Node 1: Analyze customer query and CREATE todos (write to file)
    This is the 'write_todos' phase - planning before execution
    """
    print("\n" + "="*70)
    print("🎯 PHASE 1: PLANNING SUPPORT ACTIONS")
    print("="*70)
    print(f"Ticket ID: {state['ticket_id']}")
    print(f"Customer Query: {state['customer_query']}\n")
    
    prompt = f"""
    You are a customer support agent. A customer has submitted this query:
    "{state['customer_query']}"
    
    Create a step-by-step action plan to resolve this issue. Each step should be specific and actionable.
    Consider these possible actions:
    1. Search internal knowledge base for relevant articles
    2. Look up recent similar tickets or issues
    3. Check for product updates or known bugs
    4. Gather additional context from external sources
    
    Return a numbered list of 2-4 specific actions. Be concise.
    Example format:
    1. Search knowledge base for "refund policy"
    2. Check recent complaints about payment processing
    3. Look up current refund processing times
    """
    
    response = llm.invoke([SystemMessage(content=prompt)])
    
    # Parse todos from LLM response
    todos = []
    for i, line in enumerate(response.content.strip().split('\n'), 1):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            task = line.split('.', 1)[-1].strip() if '.' in line else line.strip('- ')
            todos.append({
                "id": i,
                "task": task,
                "status": "pending",
                "result": ""
            })
    
    # WRITE TODOS TO FILE (Claude-style)
    todos_file = f"core_concepts/todos/todos_{state['ticket_id']}_{datetime.now().microsecond}.json"
    write_todos_to_file(todos, todos_file)
    
    print("\n📋 Created Action Plan:")
    for todo in todos:
        print(f"   {todo['id']}. {todo['task']}")
    
    state['todos_file'] = todos_file
    state['todos'] = todos
    state['current_todo_index'] = 0
    
    return state


def execute_support_action(state: SupportAgentState) -> SupportAgentState:
    """
    Node 2: READ todos from file and execute current one
    This demonstrates reading state from file before execution
    """
    # READ TODOS FROM FILE (Claude-style)
    todos = read_todos_from_file(state['todos_file'])
    state['todos'] = todos
    
    current_idx = state['current_todo_index']
    current_todo = todos[current_idx]
    
    print("\n" + "="*70)
    print(f"⚙️  PHASE 2: EXECUTING ACTION {current_idx + 1}/{len(todos)}")
    print("="*70)
    print(f"Task: {current_todo['task']}\n")
    
    task_lower = current_todo['task'].lower()
    result = ""
    
    # Determine action type and execute
    if "knowledge base" in task_lower or "kb" in task_lower:
        # Extract search term
        search_term = task_lower.split("for")[-1].strip().strip('"\'')
        result = search_knowledge_base(search_term, state['knowledge_base'])
        print(f"🔍 KB Search: {result[:100]}...")
        
    elif "search" in task_lower or "look up" in task_lower or "check" in task_lower:
        # Extract search query
        if "for" in task_lower:
            search_query = task_lower.split("for")[-1].strip().strip('"\'')
        else:
            search_query = task_lower.replace("search", "").replace("look up", "").replace("check", "").strip()
        
        search_results = search_ticket_database(search_query, max_results=2)
        
        if "error" in search_results:
            result = f"Error: {search_results['error']}"
        else:
            results_list = search_results.get('results', [])
            result = f"Found {len(results_list)} similar past tickets:\n"
            for r in results_list[:2]:
                result += f"- Ticket {r.get('ticket_id', 'N/A')}: {r.get('title', 'N/A')}\n"
                result += f"  {r.get('content', '')[:150]}...\n"
                result += f"  Resolved in: {r.get('resolution_time', 'N/A')}\n"
        
        print(f"🔍 Ticket DB Search: Found {len(results_list)} similar tickets")
    else:
        result = f"Completed: {current_todo['task']}"
        print(f"✅ Generic action completed")
    
    # UPDATE TODO IN FILE (Claude-style)
    update_todo_in_file(state['todos_file'], current_todo['id'], "completed", result)
    
    # Update state
    current_todo['status'] = "completed"
    current_todo['result'] = result
    state['current_todo_index'] += 1
    
    return state


def should_continue_execution(state: SupportAgentState) -> str:
    """Router: Check if more actions needed"""
    if state['current_todo_index'] < len(state['todos']):
        return "execute"
    else:
        return "synthesize"


def synthesize_resolution(state: SupportAgentState) -> SupportAgentState:
    """
    Node 3: READ all completed todos from file and create final resolution
    """
    print("\n" + "="*70)
    print("🎯 PHASE 3: SYNTHESIZING RESOLUTION")
    print("="*70)
    
    # READ FINAL STATE FROM FILE
    todos = read_todos_from_file(state['todos_file'])
    
    # Compile all results
    all_findings = "\n\n".join([
        f"Action {todo['id']}: {todo['task']}\nResult: {todo['result']}"
        for todo in todos
    ])
    
    prompt = f"""
    Customer Query: {state['customer_query']}
    
    Based on these support actions and their results:
    {all_findings}
    
    Provide a clear, helpful response to the customer. Include:
    1. Acknowledgment of their issue
    2. What you found
    3. Concrete next steps or resolution
    
    Keep it professional but friendly, under 150 words.
    """
    
    response = llm.invoke([SystemMessage(content=prompt)])
    state['final_resolution'] = response.content
    
    print("✨ Resolution generated!\n")
    
    return state


# ============================================================================
# STEP 5: Build the Graph
# ============================================================================

def create_support_agent():
    """Create and compile the support agent graph"""
    
    workflow = StateGraph(SupportAgentState)
    
    # Add nodes
    workflow.add_node("plan", plan_support_actions)
    workflow.add_node("execute", execute_support_action)
    workflow.add_node("synthesize", synthesize_resolution)
    
    # Define flow
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "execute")
    
    workflow.add_conditional_edges(
        "execute",
        should_continue_execution,
        {
            "execute": "execute",
            "synthesize": "synthesize"
        }
    )
    
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()


# ============================================================================
# STEP 6: Run Example
# ============================================================================

def main():
    """Run a customer support example"""
    
    # Simulated knowledge base
    knowledge_base = {
        "refund policy": "Refunds are processed within 5-7 business days. Eligible for items returned within 30 days.",
        "shipping": "Standard shipping takes 3-5 business days. Express shipping available for 2-day delivery.",
        "account issues": "Password resets can be done via email. For account deletion, contact support@company.com",
        "payment": "We accept all major credit cards, PayPal, and Apple Pay. Payment processing is secure via Stripe."
    }
    
    agent = create_support_agent()
    
    # Example customer query
    customer_query = "I ordered a product 10 days ago but haven't received it yet. Can I get a refund?"
    ticket_id = "TICKET-2024-001"
    
    print("\n" + "="*70)
    print("🎫 CUSTOMER SUPPORT AGENT - FILE-BASED TODO DEMO")
    print("="*70)
    
    initial_state = {
        "ticket_id": ticket_id,
        "customer_query": customer_query,
        "todos_file": "",
        "todos": [],
        "current_todo_index": 0,
        "knowledge_base": knowledge_base,
        "final_resolution": ""
    }
    
    result = agent.invoke(initial_state)
    
    print("\n" + "="*70)
    print("📧 FINAL CUSTOMER RESPONSE")
    print("="*70)
    print(f"\n{result['final_resolution']}\n")
    
    print("\n💡 Check the generated file:", result['todos_file'])
    print("   This file contains the complete action plan and results!")


if __name__ == "__main__":
    main()
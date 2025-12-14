"""
Memory-Augmented Planning - RAG-based Approach
Uses past experiences stored in memory to enhance planning.
"""

from typing import TypedDict, List, Annotated, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import os
from datetime import datetime

os.environ["OPENAI_API_KEY"] = "your-api-key-here"

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

class Memory(TypedDict):
    task: str
    plan: str
    outcome: str
    timestamp: str
    success: bool

class MemoryAugmentedState(TypedDict):
    task: str
    messages: Annotated[List, operator.add]
    memory_store: object  # FAISS vector store
    retrieved_memories: List[Memory]
    plan: str
    execution_result: str

def initialize_memory_store(state: MemoryAugmentedState) -> MemoryAugmentedState:
    """Initialize the memory store with some example past experiences."""
    embeddings = OpenAIEmbeddings()
    
    # Sample memories (in practice, these would be from actual past experiences)
    sample_memories = [
        Memory(
            task="Plan a product launch event",
            plan="1. Choose venue, 2. Send invitations, 3. Prepare demo, 4. Arrange catering, 5. Follow-up",
            outcome="Successful event with 150 attendees. Venue was slightly too small.",
            timestamp="2024-01-15",
            success=True
        ),
        Memory(
            task="Organize team offsite retreat",
            plan="1. Survey team preferences, 2. Book location, 3. Plan activities, 4. Arrange transport",
            outcome="Great feedback. Should have booked earlier for better rates.",
            timestamp="2024-02-20",
            success=True
        ),
        Memory(
            task="Launch social media campaign",
            plan="1. Create content calendar, 2. Design graphics, 3. Schedule posts, 4. Monitor engagement",
            outcome="Good engagement but needed more video content. Graphics performed best.",
            timestamp="2024-03-10",
            success=True
        ),
        Memory(
            task="Implement new software system",
            plan="1. Requirements gathering, 2. Vendor selection, 3. Installation, 4. Training",
            outcome="Failed initially due to inadequate training. Required additional support phase.",
            timestamp="2024-01-30",
            success=False
        ),
        Memory(
            task="Conduct customer survey",
            plan="1. Design questionnaire, 2. Select sample, 3. Distribute survey, 4. Analyze results",
            outcome="Low response rate. Incentives would have helped participation.",
            timestamp="2024-02-05",
            success=False
        )
    ]
    
    # Create documents for vector store
    documents = []
    for mem in sample_memories:
        content = f"Task: {mem['task']}\nPlan: {mem['plan']}\nOutcome: {mem['outcome']}"
        metadata = {
            "task": mem["task"],
            "success": mem["success"],
            "timestamp": mem["timestamp"]
        }
        documents.append(Document(page_content=content, metadata=metadata))
    
    # Create FAISS vector store
    vector_store = FAISS.from_documents(documents, embeddings)
    
    return {
        **state,
        "memory_store": vector_store
    }

def retrieve_relevant_memories(state: MemoryAugmentedState) -> MemoryAugmentedState:
    """Retrieve relevant past experiences from memory."""
    if state["memory_store"] is None:
        return {**state, "retrieved_memories": []}
    
    # Search for similar past tasks
    query = f"Task: {state['task']}"
    docs = state["memory_store"].similarity_search(query, k=3)
    
    retrieved = []
    for doc in docs:
        retrieved.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })
    
    return {
        **state,
        "retrieved_memories": retrieved
    }

def plan_with_memory(state: MemoryAugmentedState) -> MemoryAugmentedState:
    """Generate plan using insights from retrieved memories."""
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    # Format retrieved memories
    memory_context = ""
    if state["retrieved_memories"]:
        memory_context = "\n\nRelevant Past Experiences:\n"
        for i, mem in enumerate(state["retrieved_memories"], 1):
            memory_context += f"\n{i}. {mem['content']}\n"
            if mem['metadata'].get('success'):
                memory_context += "   Status: SUCCESS\n"
            else:
                memory_context += "   Status: FAILED (learn from mistakes)\n"
    
    system_msg = SystemMessage(content="""You are an experienced planner with access to past experiences.
Use insights from previous similar tasks to create a better plan.
Learn from both successes and failures.""")
    
    task_msg = HumanMessage(content=f"""Current Task: {state['task']}
{memory_context}

Based on the current task and relevant past experiences:
1. Identify what worked well in the past
2. Avoid mistakes made previously
3. Incorporate best practices
4. Create an optimized plan

Generate a comprehensive step-by-step plan.""")
    
    response = llm.invoke([system_msg, task_msg])
    
    return {
        **state,
        "plan": response.content,
        "messages": [system_msg, task_msg, response]
    }

def execute_and_store(state: MemoryAugmentedState) -> MemoryAugmentedState:
    """Simulate execution and store result in memory for future use."""
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    execution_prompt = f"""Simulate executing this plan and provide results:

Task: {state['task']}

Plan:
{state['plan']}

Provide:
1. What happened during execution
2. What worked well
3. What could be improved
4. Overall outcome"""
    
    response = llm.invoke([HumanMessage(content=execution_prompt)])
    execution_result = response.content
    
    # Store this experience in memory for future use
    embeddings = OpenAIEmbeddings()
    new_memory_content = f"Task: {state['task']}\nPlan: {state['plan']}\nOutcome: {execution_result}"
    new_doc = Document(
        page_content=new_memory_content,
        metadata={
            "task": state["task"],
            "timestamp": datetime.now().strftime("%Y-%m-%d"),
            "success": "success" in execution_result.lower()
        }
    )
    
    # Add to vector store
    if state["memory_store"]:
        state["memory_store"].add_documents([new_doc])
    
    return {
        **state,
        "execution_result": execution_result
    }

def create_memory_augmented_graph():
    """Build the memory-augmented planning workflow."""
    workflow = StateGraph(MemoryAugmentedState)
    
    # Add nodes
    workflow.add_node("init_memory", initialize_memory_store)
    workflow.add_node("retrieve", retrieve_relevant_memories)
    workflow.add_node("plan", plan_with_memory)
    workflow.add_node("execute", execute_and_store)
    
    # Add edges
    workflow.set_entry_point("init_memory")
    workflow.add_edge("init_memory", "retrieve")
    workflow.add_edge("retrieve", "plan")
    workflow.add_edge("plan", "execute")
    workflow.add_edge("execute", END)
    
    return workflow.compile()

# Example usage
if __name__ == "__main__":
    graph = create_memory_augmented_graph()
    
    initial_state = {
        "task": "Organize a conference for 200 tech professionals with keynote speakers and workshops",
        "messages": [],
        "memory_store": None,
        "retrieved_memories": [],
        "plan": "",
        "execution_result": ""
    }
    
    print("Starting Memory-Augmented Planning Agent...")
    print(f"Task: {initial_state['task']}\n")
    
    result = graph.invoke(initial_state)
    
    print("=== RETRIEVED MEMORIES ===")
    for i, mem in enumerate(result["retrieved_memories"], 1):
        print(f"\nMemory {i}:")
        print(mem["content"][:300] + "..." if len(mem["content"]) > 300 else mem["content"])
        print(f"Status: {'SUCCESS' if mem['metadata'].get('success') else 'FAILED'}")
    
    print("\n\n=== GENERATED PLAN (WITH MEMORY INSIGHTS) ===")
    print(result["plan"])
    
    print("\n\n=== EXECUTION RESULT ===")
    print(result["execution_result"])
    
    print("\n\n[Note: This experience has been stored in memory for future tasks]")
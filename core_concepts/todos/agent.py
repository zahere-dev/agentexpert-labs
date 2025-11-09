# langgraph_todos_tool_decorator.py
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from dotenv import load_dotenv
load_dotenv()

# -------------------------------------------------
# 1. CONFIG
# -------------------------------------------------
DATA_DIR = Path("./todos_data")
DATA_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# 2. Todo Schema
# -------------------------------------------------
class TodoItem(BaseModel):
    id: str = Field(default_factory=lambda: f"todo-{uuid.uuid4().hex[:8]}")
    description: str
    validation: str
    status: str = "pending"
    evidence: Optional[str] = None
    error_message: Optional[str] = None
    version: int = 1
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    priority: int = 0
    tags: List[str] = Field(default_factory=list)

    @field_validator("status")
    def valid_status(cls, v):
        if v not in {"pending", "in_progress", "done", "failed"}:
            raise ValueError("Invalid status")
        return v

# -------------------------------------------------
# 3. Agent State (with file persistence)
# -------------------------------------------------
class AgentState(BaseModel):
    user_id: str
    session_id: str
    goal: str
    todos: List[TodoItem] = Field(default_factory=list)
    current_todo_index: int = 0
    phase: Literal["planning", "execution", "done"] = "planning"
    messages: List[dict] = Field(default_factory=list)

    @property
    def file_path(self) -> Path:
        safe_user = "".join(c if c.isalnum() else "_" for c in self.user_id)
        safe_sess = "".join(c if c.isalnum() else "_" for c in self.session_id)
        return DATA_DIR / f"{safe_user}__{safe_sess}.json"

    def save(self):
        """Persist state to JSON file."""
        with open(self.file_path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load(cls, user_id: str, session_id: str) -> Optional["AgentState"]:
        """Load state from file if exists."""
        path = cls(user_id=user_id, session_id=session_id, goal="", todos=[]).file_path
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

# -------------------------------------------------
# 4. Tools with @tool + Docstrings
# -------------------------------------------------
@tool
def read_todos(state: AgentState) -> List[Dict[str, Any]]:
    """
    Read the current todo list with full status, evidence, and version.

    Returns:
        List of todo dicts (id, description, status, evidence, etc.)
    """
    return [t.model_dump() for t in state.todos]


@tool
def write_todos(state: AgentState, todos: List[dict], phase: str) -> AgentState:
    """
    Generate or update the todo list.

    Args:
        todos: List of todo dicts (with id, description, validation, etc.)
        phase: 'planning' or 'execution'

    Behavior:
        - If id exists → merge (preserve evidence, increment version)
        - If new → append
        - Auto-saves to file
    """
    existing = {t.id: t for t in state.todos}
    updated = []

    for t in todos:
        todo_id = t.get("id")
        if todo_id and todo_id in existing:
            old = existing[todo_id]
            merged = old.model_copy(update={k: v for k, v in t.items() if v is not None})
            merged.version = old.version + 1
            merged.updated_at = datetime.utcnow().isoformat()
            updated.append(merged)
        else:
            new_todo = TodoItem(**{k: v for k, v in t.items() if v is not None})
            updated.append(new_todo)

    state.todos = updated
    state.phase = phase
    state.current_todo_index = 0
    state.save()
    return state


@tool
def execute_code(state: AgentState, code: str) -> str:
    """
    Execute Python code in a safe sandbox.

    Args:
        code: Python code string. Should set `result = ...` on success.

    Returns:
        SUCCESS: <result> or ERROR: <message>
    """
    try:
        local = {}
        exec(code, {}, local)
        result = local.get("result", "No result variable")
        return f"SUCCESS: {result}"
    except Exception as e:
        return f"ERROR: {str(e)}"


# -------------------------------------------------
# 5. LLM with @tool-decorated functions
# -------------------------------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools([read_todos, write_todos, execute_code])

# -------------------------------------------------
# 6. Nodes
# -------------------------------------------------
def planner_node(state: AgentState):
    current = json.dumps(read_todos(state), indent=2)
    prompt = f"""
Goal: {state.goal}

CURRENT TODOS (never repeat done tasks):
{current or '[]'}

Break into 3–7 verifiable subtasks. Use write_todos(phase="planning").
Only add new or update incomplete ones.
"""
    response = llm_with_tools.invoke([HumanMessage(content=prompt)] + state.messages)
    state.messages.append(response)
    return state


def executor_node(state: AgentState):
    if state.current_todo_index >= len(state.todos):
        state.phase = "done"
        state.save()
        return state

    todo = state.todos[state.current_todo_index]
    if todo.status in {"done", "failed"}:
        state.current_todo_index += 1
        return state

    current = json.dumps(read_todos(state), indent=2)
    prompt = f"""
Execute next task:

CURRENT STATE:
{current}

TASK:
Description: {todo.description}
Validation: {todo.validation}

Write Python code. Set `result = ...` on success.
"""
    response = llm_with_tools.invoke([HumanMessage(content=prompt)] + state.messages)
    state.messages.append(response)

    for tool_call in (response.tool_calls or []):
        if tool_call["name"] == "execute_code":
            output = execute_code(state, tool_call["args"]["code"])
            todo.status = "done" if output.startswith("SUCCESS") else "failed"
            todo.evidence = output
            todo.version += 1
            todo.updated_at = datetime.utcnow().isoformat()
            state.current_todo_index += 1
            state = write_todos(state, [todo.model_dump()], "execution")
        elif tool_call["name"] == "write_todos":
            args = tool_call["args"]
            state = write_todos(state, args["todos"], args["phase"])

    return state


# -------------------------------------------------
# 7. Graph
# -------------------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("tools", ToolNode(tools=[read_todos, write_todos, execute_code]))

workflow.add_edge(START, "planner")
workflow.add_conditional_edges("planner", tools_condition)
workflow.add_conditional_edges("executor", tools_condition)

app = workflow.compile()

# -------------------------------------------------
# 8. Run Helper
# -------------------------------------------------
async def run_agent(goal: str, user_id: str = "alice", session_id: str = "deploy1"):
    thread_id = f"{user_id}:{session_id}"
    config = {"configurable": {"thread_id": thread_id}}

    state = AgentState.load(user_id, session_id)
    if state is None:
        state = AgentState(user_id=user_id, session_id=session_id, goal=goal)
    else:
        print(f"Resuming session {thread_id} with {len(state.todos)} todos")

    print(f"\nGoal: {goal}\n" + "="*60)
    async for event in app.astream(state.model_dump(), config):
        s = AgentState(**event.get("executor") or event.get("planner") or event)
        if s.todos:
            print(f"PHASE: {s.phase.upper()}")
            for t in s.todos:
                status = "DONE" if t.status == "done" else "PENDING" if t.status == "pending" else t.status.upper()
                print(f"  [{status}] {t.description}")
                if t.evidence:
                    print(f"     Evidence: {t.evidence[:100]}")
            print()


# -------------------------------------------------
# 9. Run Example
# -------------------------------------------------
if __name__ == "__main__":
    import asyncio
    goal = "Create a FastAPI app that says 'Hello AI' on / and save it to hello_app/"
    asyncio.run(run_agent(goal, user_id="alice", session_id="fastapi-1"))
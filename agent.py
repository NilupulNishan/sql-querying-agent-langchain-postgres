# ============================================================
# agent.py
# LangGraph agent:
#   1. Build schema from PostgreSQL
#   2. Ask LLM to write Python code as the plan
#   3. Extract <execute_python> block
#   4. Run code safely in a sandbox
#   5. Return answer_text
#
# This is READ-ONLY: the sandbox only allows SELECT queries.
# ============================================================

from __future__ import annotations
 
import io
import os
import re
import sys
import traceback
from typing import Any, Optional, TypedDict
 
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
 
import db_utils

load_dotenv()

# ============================================================
# 1. LangGraph State
# ============================================================
 
class AgentState(TypedDict):
    # Input
    question: str                    # The user's natural language question
 
    # Built by schema_node
    schema_block: str                # Text description of the DB tables
 
    # Built by generate_code_node
    raw_llm_response: str            # Full LLM response (with <execute_python> tags)
 
    # Built by execute_code_node
    extracted_code: str              # Just the Python between the tags
    stdout_log: str                  # Any print() output from the generated code
    execution_error: Optional[str]   # Traceback if code crashed, else None
    answer_text: Optional[str]       # The variable set by the generated code
    status: Optional[str]            # success / no_match / insufficient_stock / etc.
 
    # Final output
    final_answer: str                # What we show the user


# ============================================================
# 2. The Prompt
# ============================================================
 
PROMPT_TEMPLATE = """You are a senior data assistant for a sunglasses store. 
PLAN BY WRITING PYTHON CODE USING PSYCOPG2 to query a READ-ONLY PostgreSQL database.
 
Database Schema & Samples (read-only):
{schema_block}
 
Execution Environment (already available in the sandbox):
- Variable: conn  (a psycopg2 read-only connection — already open)
- Use: conn.cursor() or psycopg2.extras.RealDictCursor for dict rows
- Standard library (re, datetime) is available
- This is READ-ONLY. NEVER write INSERT, UPDATE, DELETE, or DDL.
 
PLANNING RULES (critical):
- Derive ALL filters/parameters from user_request (shape/keywords, price ranges "under/over/between",
  stock mentions, quantities, buy/return intent). Do NOT hard-code values.
- Build SQL queries dynamically using parameterized queries (%s placeholders).
- Be conservative: if intent is ambiguous, do a read-only query only.
 
HUMAN RESPONSE REQUIREMENT (hard):
- You MUST set a variable named `answer_text` (type str) with a short, customer-friendly sentence (1-2 lines).
- If nothing matches, politely say so and offer the nearest alternative (closest style/price).
 
ACTION POLICY:
- Since this is read-only, ACTION is always "read". SHOULD_MUTATE is always False.
- Still simulate what WOULD happen for purchase/return requests and describe it in answer_text.
 
FAILURE & EDGE-CASE HANDLING (must implement):
- Always set a short `answer_text` and a string `STATUS` to one of:
  "success", "no_match", "insufficient_stock", "invalid_request", "unsupported_intent"
- no_match: No items satisfy the filters → suggest the closest in style/price.
- insufficient_stock: Item found but stock < requested qty → state available qty.
- invalid_request: Unable to parse essential info → ask for the missing piece.
- In all cases, keep tone helpful and concise (1-2 sentences).
 
OUTPUT CONTRACT:
- Return ONLY executable Python between these tags (no extra text):
  <execute_python>
  # your python code here
  </execute_python>
 
CODE CHECKLIST (follow in code):
1) Parse intent & constraints from user_request (regex ok).
2) Build a SQL query dynamically with %s parameters — never f-strings with user data.
3) Execute query using the provided conn object.
4) ALWAYS set:
   - `answer_text` (human sentence, required)
   - `STATUS` (see list above)
   Also print: LOG: ACTION=read STATUS=<status>
5) Use psycopg2.extras.RealDictCursor so rows are dicts.
 
SQL EXAMPLES (use these patterns):
  with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
      cur.execute("SELECT * FROM inventory WHERE price < %s AND quantity_in_stock > 0", (100,))
      rows = cur.fetchall()
 
TONE EXAMPLES (for answer_text):
- success: "Yes, we have our Classic sunglasses, a round frame, for $60."
- no_match: "We don't have round frames under $100 right now, but our Moon round frame is $120."
- insufficient_stock: "We only have 1 pair of Classic left; I can reserve that for you."
- invalid_request: "I can help—how many pairs would you like?"
 
User request:
{question}
"""

 
# ============================================================
# 3. LLM setup — Azure OpenAI
# ============================================================
 
def get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_GPT4O_DEPLOYMENT"),
        temperature=0.2,
    )


# ============================================================
# 4. _extract_execute_block
# ============================================================
 
def _extract_execute_block(text: str) -> str:
    """
    Returns the Python code inside <execute_python>...</execute_python>.
    If no tags are found, assumes the entire text is raw Python.
    Mirrors the notebook's _extract_execute_block exactly.
    """
    if not text:
        raise RuntimeError("Empty content passed to code extractor.")
    m = re.search(
        r"<execute_python>(.*?)</execute_python>",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else text.strip()


# ============================================================
# 5. execute_generated_code — same as notebook, but sandbox
#    uses psycopg2 conn instead of TinyDB tables
# ============================================================
 
def execute_generated_code(
    code_or_content: str,
    *,
    conn: psycopg2.extensions.connection,
    user_request: str = "",
) -> dict[str, Any]:
    """
    Execute LLM-generated code in a controlled sandbox.
    Accepts either raw Python OR full LLM content with <execute_python> tags.
 
    The sandbox provides:
      - conn: read-only psycopg2 connection
      - psycopg2.extras: for RealDictCursor
      - re, datetime: standard library helpers
      - user_request: the original question string
 
    Returns:
      {
        "code":         extracted Python string,
        "stdout":       captured print() output,
        "error":        traceback string or None,
        "answer_text":  the answer_text variable set by the code,
        "status":       the STATUS variable set by the code,
      }
    """
    # Extract just the code portion
    code = _extract_execute_block(code_or_content)
 
    # ---- Safe sandbox globals ----
    # Only expose what the generated code needs.
    # __builtins__ is restricted to prevent imports of dangerous modules.
    import re as _re
    import datetime as _datetime
 
    SAFE_GLOBALS = {
        "__builtins__": {
            # Basic builtins the code might need
            "print": print,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "sorted": sorted,
            "any": any,
            "all": all,
            "isinstance": isinstance,
            "None": None,
            "True": True,
            "False": False,
        },
        # Libraries the generated code needs
        "re": _re,
        "datetime": _datetime,
        "psycopg2": psycopg2,
        # Convenience shortcut so the LLM can write:
        # with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        "RealDictCursor": psycopg2.extras.RealDictCursor,
    }
 
    SAFE_LOCALS = {
        "conn": conn,
        "user_request": user_request,
    }
 
    # ---- Capture stdout ----
    stdout_buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_buf
    error_text = None
 
    try:
        exec(code, SAFE_GLOBALS, SAFE_LOCALS)  # noqa: S102
    except Exception:
        error_text = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
 
    captured_stdout = stdout_buf.getvalue().strip()
 
    return {
        "code":        code,
        "stdout":      captured_stdout,
        "error":       error_text,
        "answer_text": SAFE_LOCALS.get("answer_text"),
        "status":      SAFE_LOCALS.get("STATUS", "unknown"),
    }


# ============================================================
# 6. LangGraph Nodes
# Each node is a plain Python function that receives the state,
# does one thing, and returns a dict of updates to the state.
# ============================================================
def schema_node(state: AgentState) -> dict:
    """Node 1: Connect to PostgreSQL and build the schema description."""

    print("[schema_node] Connecting to PostgreSQL and building schema...")
    conn = db_utils.get_connection()
    schema = db_utils.build_schema_block(conn)
    conn.close()
    print("[schema_node] Schema built successfully.")
    return {"schema_block": schema}


def generate_code_node(state: AgentState) -> dict:
    """Node 2: Ask the LLM to write Python code as the plan."""

    print("[generate_code_node] Asking LLM to generate plan-as-code...")
    llm = get_llm()
 
    prompt = PROMPT_TEMPLATE.format(
        schema_block=state["schema_block"],
        question=state["question"],
    )
 
    # LangChain message format
    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [
        SystemMessage(content="You write safe, well-commented psycopg2 Python code to answer data questions."),
        HumanMessage(content=prompt),
    ]
 
    response = llm.invoke(messages)
    raw_content = response.content or ""
    print("[generate_code_node] LLM response received.")
    print("--- RAW LLM RESPONSE ---")
    print(raw_content)
    print("------------------------")
    return {"raw_llm_response": raw_content}


def execute_code_node(state: AgentState) -> dict:
    """
    Node 3: Extract the <execute_python> block and run it safely.
    Mirrors execute_generated_code() from the notebook, but uses PostgreSQL.
    """
    print("[execute_code_node] Extracting and executing generated code...")
 
    # Open a fresh read-only connection for the sandbox
    conn = db_utils.get_connection()
 
    try:
        result = execute_generated_code(
            state["raw_llm_response"],
            conn=conn,
            user_request=state["question"],
        )
    finally:
        conn.close()
 
    print(f"[execute_code_node] stdout: {result['stdout']}")
    if result["error"]:
        print(f"[execute_code_node] ERROR:\n{result['error']}")
 
    return {
        "extracted_code": result["code"],
        "stdout_log":     result["stdout"],
        "execution_error": result["error"],
        "answer_text":    result["answer_text"],
        "status":         result["status"],
    }


def answer_node(state: AgentState) -> dict:
    """
    Node 4: Compose the final answer shown to the user.
    If the code crashed, return a friendly fallback message.
    """
    if state.get("execution_error"):
        final = (
            "Sorry, I ran into a problem processing your request. "
            "Please try rephrasing your question."
        )
        print(f"[answer_node] Execution error detected, returning fallback.")
        print(f"[answer_node] Error was:\n{state['execution_error']}")
    elif state.get("answer_text"):
        final = state["answer_text"]
    else:
        final = "I couldn't find a clear answer. Could you rephrase your question?"
 
    print(f"[answer_node] Final answer: {final}")
    return {"final_answer": final}


# ============================================================
# 7. Build the LangGraph graph
# ============================================================
 
def build_agent_graph() -> Any:
    """
    Wire up the four nodes into a LangGraph StateGraph.
    Flow: schema → generate_code → execute_code → answer → END
    """
    # Create graph
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("schema_node", schema_node)
    graph.add_node("generate_code_node", generate_code_node)
    graph.add_node("execute_code_node", execute_code_node)
    graph.add_node("answer_node", answer_node)

    # Define edges
    graph.set_entry_point("schema_node")
    graph.add_edge("schema_node", "generate_code_node")
    graph.add_edge("generate_code_node", "execute_code_node")
    graph.add_edge("execute_code_node", "answer_node")
    graph.add_edge("answer_node", END)

    return graph.compile()


# ============================================================
# 8. Public function — run a single question through the agent
# ============================================================
 
def run_agent(question: str) -> str:
    """
    Main entry point. Takes a natural language question,
    runs it through the full graph, returns the answer string.
    """
    app = build_agent_graph()
 
    initial_state: AgentState = {
        "question":          question,
        "schema_block":      "",
        "raw_llm_response":  "",
        "extracted_code":    "",
        "stdout_log":        "",
        "execution_error":   None,
        "answer_text":       None,
        "status":            None,
        "final_answer":      "",
    }
 
    final_state = app.invoke(initial_state)
    return final_state["final_answer"]


import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Ask your question: ")

    print(run_agent(question))
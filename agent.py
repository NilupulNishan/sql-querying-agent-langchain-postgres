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
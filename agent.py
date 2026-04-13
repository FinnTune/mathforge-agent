# agent.py
from langgraph.prebuilt import create_react_agent
from langchain_xai import ChatXAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate   # ← NEW IMPORT
from dotenv import load_dotenv
from tools import execute_python_code

load_dotenv()

# Grok via xAI
llm = ChatXAI(
    model="grok-4",
    temperature=0
)

tools = [execute_python_code]

# System prompt (kept exactly the same)
system_message = SystemMessage(
    content="""You are MathForge, an expert mathematician and Python coder powered by Grok.
    Your job is to solve math and coding problems using clear reasoning.
    Always:
    1. Think step-by-step.
    2. Write clean, correct Python code.
    3. Execute it with the execute_python_code tool.
    4. Verify the result.
    5. Give a friendly, educational final answer with explanations.
    Use SymPy for symbolic math, NumPy/SciPy for numerics, Matplotlib for plots.
    Never guess — always execute code to confirm."""
)

# Convert to the new prompt format that LangGraph expects
prompt = ChatPromptTemplate.from_messages([system_message])

# Create the ReAct agent (updated for 2026 LangGraph)
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=prompt,      # ← This is the new correct parameter
    debug=False
)

print("✅ MathForge agent loaded successfully with Grok!")  # optional helpful message

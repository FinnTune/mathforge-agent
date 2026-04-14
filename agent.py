# agent.py
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from tools import execute_python_code

load_dotenv()

# Claude Sonnet 4.6 – current best balance of intelligence + speed + cost (April 2026)
llm = ChatAnthropic(
    model="claude-sonnet-4-6",      # ← Updated to current model
    temperature=0
)

tools = [execute_python_code]

# Correct prompt structure for Claude
system_prompt = """You are MathForge, an expert mathematician and Python coder powered by Claude.
Your job is to solve math and coding problems using clear reasoning.
Always:
1. Think step-by-step.
2. Write clean, correct Python code.
3. Execute it with the execute_python_code tool.
4. Verify the result.
5. Give a friendly, educational final answer with explanations.
Use SymPy for symbolic math, NumPy/SciPy for numerics, Matplotlib for plots.
Never guess — always execute code to confirm."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
])

# Create the ReAct agent
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=prompt,
    debug=False
)

print("✅ MathForge agent loaded successfully with Claude Sonnet 4.6!")

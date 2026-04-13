# tools.py
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_core.tools import tool
import os

# Safe sandboxed Python REPL with math libraries
repl_tool = PythonREPLTool()

@tool
def execute_python_code(code: str) -> str:
    """Execute Python code in a REPL. Use this for ANY math or coding task.
    Pre-imported: sympy, numpy, matplotlib, scipy.
    For plots: save to './plots/figure.png' and describe it.
    Returns output or printed results."""
    os.makedirs("plots", exist_ok=True)
    result = repl_tool.run(code)
    return f"Execution result:\n{result}\n(Plots saved to ./plots/ if generated)"

# 🧪 MathForge – LangGraph + Grok Math & Code Agent

A ReAct AI agent built with **LangGraph** and **Grok (xAI)** that solves math/coding problems by writing and safely executing real Python code.

## Features
- Powered by Grok-4 (excellent reasoning + tool calling)
- Symbolic & numerical math
- Automatic plotting (images saved locally)
- Full step-by-step explanations

## Quick Start (100% secure)
```bash
git clone https://github.com/yourusername/mathforge-agent.git
cd mathforge-agent
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
# IMPORTANT: Never commit your API key
`cp .env.example .env`
# → Edit .env and add your xAI key from https://console.x.ai/
`python main.py`

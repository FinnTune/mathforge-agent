# main.py
from agent import agent
from langchain_core.messages import HumanMessage
import asyncio

async def main():
    print("🧪 Welcome to MathForge (powered by Claude)! Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        
        print("Thinking...\n")
        result = await agent.ainvoke({"messages": [HumanMessage(content=query)]})
        final_message = result["messages"][-1].content
        print(f"MathForge: {final_message}\n")

if __name__ == "__main__":
    asyncio.run(main())

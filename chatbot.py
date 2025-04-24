from langchain_community.llms import Ollama  
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.tools import Tool
import pandas as pd

def ollama_agent(question, df: pd.DataFrame):
    tools = [
        Tool(
            name="Get Data Summary",
            func=lambda q: df.describe().to_string(),
            description="Useful for summarizing a dataset."
        )
    ]

    llm = Ollama(model="phi")

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  
        verbose=False
    )

    try:
        return agent.run(question)
    except Exception as e:
        return f"❌ Error: {str(e)}"

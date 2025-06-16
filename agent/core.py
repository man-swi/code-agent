from typing import List
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.memory import ConversationBufferMemory

# Import your modules
from tools.python_executor import PythonCodeExecutorTool
from tools.task_completed import TaskCompletedTool 
from agent.prompt import prompt
from agent.llm_config import get_groq_llm

def get_agent_executor():
    """Initializes and returns the LangChain Agent Executor."""
    llm = get_groq_llm()
    # Add both tools to the list
    tools: List[BaseTool] = [PythonCodeExecutorTool(), TaskCompletedTool()]

    # Initialize memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create the ReAct Agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # Create the Agent Executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=20, # Set a reasonable limit to prevent infinite loops and allow reset
        memory=memory,
    )
    return agent_executor
from typing import List
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import BaseTool
from langchain.memory import ConversationBufferMemory

# Import your modules
from tools.python_executor import PythonCodeExecutorTool
from tools.task_completed import TaskCompletedTool
from agent.prompt import prompt
from agent.llm_config import get_groq_llm

def get_agent_executor():
    """
    Initializes and returns the LangChain Agent Executor.

    This function sets up the agent with the necessary components:
    - The Language Model (LLM) from Groq, configured via session state.
    - The tools the agent can use (Python Executor, Task Completed).
    - The ReAct prompt template that guides the agent's reasoning.
    - Conversation memory to maintain context.
    """
    # The get_groq_llm function now reads model and temperature
    # directly from Streamlit's session state.
    llm = get_groq_llm()

    # Define the list of tools available to the agent.
    tools: List[BaseTool] = [PythonCodeExecutorTool(), TaskCompletedTool()]

    # Initialize memory to store conversation history.
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create the ReAct Agent using the LLM, tools, and prompt.
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # Create the Agent Executor which runs the agent's reasoning loop.
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=20,  # A safety limit to prevent infinite loops.
        memory=memory,
    )
    return agent_executor
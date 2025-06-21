# agent/core.py
import os
from typing import List
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import BaseTool, render_text_description
from langchain.memory import ConversationBufferMemory

from tools.python_executor import PythonCodeExecutorTool
from tools.task_completed import TaskCompletedTool
from agent.prompt import prompt
from agent.llm_config import get_groq_llm

def get_agent_executor():
    """Initializes and returns the LangChain Agent Executor with all fixes."""
    llm = get_groq_llm()
    tools: List[BaseTool] = [PythonCodeExecutorTool(), TaskCompletedTool()]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    rendered_tools = render_text_description(tools)
    tool_names = ", ".join([t.name for t in tools])

    final_prompt = prompt.partial(
        tools=rendered_tools,
        tool_names=tool_names,
    )

    try:
        max_iterations = int(os.getenv("AGENT_MAX_ITERATIONS", "20"))
    except (ValueError, TypeError):
        max_iterations = 20

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=final_prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors="Check your output and make sure it conforms to the Action/Action Input format.",
        max_iterations=max_iterations,
        memory=memory,
    )
    return agent_executor
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

class TaskCompletedToolInput(BaseModel):
    final_answer: str = Field(description="The complete and final answer to the user's original question.")

class TaskCompletedTool(BaseTool):
    name: str = "task_completed"
    description: str = (
        "Call this tool when you have completely and definitively answered the user's original question. "
        "The input to this tool should be your complete Final Answer. "
        "Do NOT call this tool if the task is not fully complete or requires further actions or user input."
    )
    args_schema: Type[BaseModel] = TaskCompletedToolInput

    def _run(self, final_answer: str) -> str:
        """
        This tool's execution simply signifies completion.
        The Streamlit application's callback handler will intercept this.
        """
        # Return a simple message as the callback handler will intercept the action
        return "Task completion signal received."

    async def _arun(self, final_answer: str) -> str:
        return self._run(final_answer)
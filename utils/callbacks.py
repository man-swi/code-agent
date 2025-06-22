import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from typing import Any, Optional
import sys
from io import StringIO
import re

class StreamlitCodeExecutionCallbackHandler(BaseCallbackHandler):
    """
    A custom LangChain callback handler to integrate the agent with the Streamlit UI.

    This handler intercepts agent actions to implement:
    1.  **Human-in-the-Loop (HIL):** Pauses execution when the agent wants to run
        Python code, displaying it to the user for approval.
    2.  **Final Answer Display:** Catches the `task_completed` tool call to
        gracefully end the agent's turn and display the final result.
    3.  **Log Capturing:** Redirects `stdout` to capture the agent's internal
        "thought" process for display in an expander.
    """
    def __init__(self, python_executor_tool_instance: Any, task_completed_tool_instance: Any, log_buffer: StringIO):
        super().__init__()
        self.python_executor_tool = python_executor_tool_instance
        self.task_completed_tool = task_completed_tool_instance
        self.captured_logs_buffer = log_buffer
        self._original_stdout = sys.stdout

    def _strip_ansi_codes(self, s: str) -> str:
        """Removes ANSI escape codes from a string for cleaner display."""
        return re.sub(r'\x1b\[[0-9;]*m', '', s)

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """
        Intercepts agent actions before they are executed.
        """
        # Redirect stdout to the buffer to capture the agent's thought process logs.
        sys.stdout = self.captured_logs_buffer
        action_log_entry = self._strip_ansi_codes(action.log)

        # Case 1: Intercept Python code execution for user approval.
        # <<< MODIFIED: Now extracts the 'Thought' and saves it to session state >>>
        if action.tool == self.python_executor_tool.name:
            # Extract just the 'Thought:' part of the log for cleaner display
            thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:)", action_log_entry, re.DOTALL)
            thought_text = thought_match.group(1).strip() if thought_match else "The agent did not provide a thought."

            st.session_state.pending_action = {
                "tool": action.tool,
                "tool_input": action.tool_input,
                "thought": thought_text,  # Save the extracted thought
            }
            st.session_state.last_agent_action_log_entry = action_log_entry

            sys.stdout = self._original_stdout
            st.rerun() # This will stop execution and trigger the HIL UI
            raise InterceptToolCall("Intercepted python_code_executor for HIL.")

        # Case 2: Intercept the final answer to display it correctly.
        elif action.tool == self.task_completed_tool.name:
            st.session_state.pending_final_answer = action.tool_input
            st.session_state.last_agent_action_log_entry = action_log_entry
            st.session_state.agent_continuation_needed = False
            st.session_state.current_agent_chain_user_prompt = None

            sys.stdout = self._original_stdout
            st.rerun() # This will stop execution and trigger the final answer UI
            raise InterceptToolCall("Intercepted task_completed to display final answer.")

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Restores stdout after a tool has finished running."""
        sys.stdout = self._original_stdout
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Restores stdout when the agent finishes its reasoning cycle."""
        sys.stdout = self._original_stdout
        pass

class InterceptToolCall(Exception):
    """Custom exception to halt the LangChain execution chain when a tool call is intercepted."""
    pass
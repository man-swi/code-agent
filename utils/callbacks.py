# callbacks.py
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
        if not isinstance(s, str):
            return str(s) # Ensure it's a string for regex operations
        return re.sub(r'\x1b\[[0-9;]*m', '', s)

    def _extract_thought_from_log(self, log_content: str) -> str:
        """
        Attempts to extract the 'Thought:' or 'Plan:' section from the agent's log.
        Handles cases where the log might be malformed or incomplete.
        """
        if not log_content:
            return "" # Return empty string if no log content

        # Regex to find "Thought:" or "Plan:" and capture its content non-greedily
        # until the next major section marker (Action:, Tool Call:) or end of string.
        thought_match = re.search(
            r"(?:Thought:|Plan:)\s*(.*?)(?=\n(?:Action:|Tool Call:|$))", 
            log_content, 
            re.DOTALL | re.IGNORECASE
        )

        if thought_match and thought_match.group(1):
            thought_text = thought_match.group(1).strip()
            # Clean up any lines that are just whitespace or specific delimiters
            cleaned_lines = [line.strip() for line in thought_text.split('\n') if line.strip() and not line.strip().startswith('---')]
            return "\n".join(cleaned_lines)
        else:
            # Fallback: If no clear "Thought:" section, try to find any lines that aren't actions.
            lines = log_content.split('\n')
            potential_thoughts = []
            for line in lines:
                stripped_line = line.strip()
                if stripped_line and not stripped_line.lower().startswith(('action:', 'tool call:', 'action input:')):
                    potential_thoughts.append(stripped_line)
            
            if potential_thoughts:
                return "\n".join(potential_thoughts)
            else:
                return "" # Return empty string if nothing useful found

    def get_default_agent_plan(self) -> str:
        """
        Returns a default agent plan detailing understanding, strategy, code design,
        expected outcome, and assumptions. This is used when the LLM fails to
        provide a discernible thought process.
        """
        return (
            "Understanding: The agent has interpreted the request and is preparing to execute code.\n"
            "Strategy: The agent will use the provided Python code as is.\n"
            "Code Design: The code snippet is self-contained and aims to fulfill the request.\n"
            "Expected Outcome: The code will be executed, and its output will be displayed.\n"
            "Assumptions: The provided code is syntactically correct and will run as intended."
        )

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """
        Intercepts agent actions before they are executed.
        """
        # Redirect stdout to the buffer to capture the agent's thought process logs.
        sys.stdout = self.captured_logs_buffer
        
        # Clean and process the action log.
        action_log_content = self._strip_ansi_codes(str(action.log)) if action.log else "No log available."
        
        # Store the raw, cleaned log for potential detailed display later if needed.
        st.session_state.last_agent_action_log_entry = action_log_content

        # Extract the Thought/Plan from the log content.
        agent_thought_content = self._extract_thought_from_log(action_log_content)
        
        # Compare fallback texts case-insensitively.
        fallback_phrases = {
            "", 
            "the agent did not provide a thought.", 
            "no log content available to extract thought.",
            "the agent did not provide a discernible thought process."
        }
        if not agent_thought_content or agent_thought_content.strip().lower() in fallback_phrases:
            agent_thought_content = self.get_default_agent_plan()
        
        # Format the thought for display in the chat history.
        formatted_thought = f"**Agent's Plan:**\n{agent_thought_content}"
        
        # Case 1: Intercept Python code execution for user approval.
        if action.tool == self.python_executor_tool.name:
            # Force action.tool_input to a string, handling None and empty strings.
            proposed_code = str(action.tool_input) if action.tool_input is not None else "# No code was proposed."
            if not proposed_code.strip():
                proposed_code = "# No code was proposed."
            
            # >>> SAVE TO SESSION STATE <<<
            # This thought and code will be used by custom_code_agent.py to render the UI
            st.session_state.pending_action = {
                "tool": action.tool,
                "tool_input": proposed_code, # Save proposed code here
                "thought": formatted_thought, # Save the formatted thought here
            }

            # Restore stdout and then rerun the UI.
            sys.stdout = self._original_stdout
            st.rerun() 
            raise InterceptToolCall("Intercepted python_code_executor for HIL.")

        # Case 2: Intercept the final answer to display it correctly.
        elif action.tool == self.task_completed_tool.name:
            st.session_state.pending_final_answer = action.tool_input
            st.session_state.last_agent_action_log_entry = action_log_content # Save log for final thought display
            st.session_state.agent_continuation_needed = False
            st.session_state.current_agent_chain_user_prompt = None

            sys.stdout = self._original_stdout
            st.rerun()
            raise InterceptToolCall("Intercepted task_completed to display final answer.")
        
        # Restore stdout if no tool requires special handling. This is a safeguard.
        sys.stdout = self._original_stdout

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
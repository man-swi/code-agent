import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from typing import Any, Dict, List, Optional
import os
import sys
from io import StringIO
import re # Added for stripping ANSI codes

class StreamlitCodeExecutionCallbackHandler(BaseCallbackHandler):
    """
    A custom LangChain callback handler to manage Human-in-the-Loop (HIL)
    for code execution and final answer display in a Streamlit application.
    """
    def __init__(self, python_executor_tool_instance: Any, task_completed_tool_instance: Any, log_buffer: StringIO): 
        super().__init__()
        self.python_executor_tool = python_executor_tool_instance
        self.task_completed_tool = task_completed_tool_instance # NEW: Reference to TaskCompletedTool
        self.captured_logs_buffer = log_buffer 
        self._original_stdout = sys.stdout # Store original stdout

    # Helper function to strip ANSI escape codes
    def _strip_ansi_codes(self, s):
        return re.sub(r'\x1b\[[0-9;]*m', '', s)

    def on_agent_action(
        self,
        action: AgentAction,
        **kwargs: Any,
    ) -> Any:
        """
        Called when agent takes an action.
        If it's our PythonCodeExecutorTool or TaskCompletedTool, we intercept it.
        """
        # Ensure stdout is redirected before logging the action
        # This is crucial to capture agent's internal thoughts for display
        sys.stdout = self.captured_logs_buffer 
        
        # Capture the raw log entry before interception
        action_log_entry = self._strip_ansi_codes(action.log)
        
        if action.tool == self.python_executor_tool.name:
            st.session_state.pending_action = {
                "tool": action.tool,
                "tool_input": action.tool_input, 
            }
            st.session_state.last_agent_action_log_entry = action_log_entry
            
            # Temporarily restore stdout before rerunning Streamlit
            sys.stdout = self._original_stdout 
            st.rerun() 
            
            raise InterceptToolCall("Intercepted python_code_executor for Human-in-the-Loop.")
        
        elif action.tool == self.task_completed_tool.name: # NEW: Intercept TaskCompletedTool
            st.session_state.pending_final_answer = action.tool_input # Store the final answer
            st.session_state.last_agent_action_log_entry = action_log_entry # Store the thought that led to it
            
            # Reset agent state as the task is considered complete from UI perspective
            st.session_state.agent_continuation_needed = False
            st.session_state.current_agent_chain_user_prompt = None # Reset the chain's original prompt
            
            # Temporarily restore stdout before rerunning Streamlit
            sys.stdout = self._original_stdout 
            st.rerun()
            
            raise InterceptToolCall("Intercepted task_completed to display final answer.")


    def on_tool_end(
        self,
        output: str,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Called after tool ends running.
        We'll only reach here if the tool was NOT intercepted.
        Ensure stdout is restored.
        """
        sys.stdout = self._original_stdout 
        pass

    def on_agent_finish(
        self,
        finish: AgentFinish,
        **kwargs: Any,
    ) -> Any:
        """
        Called when agent finishes. Restore stdout.
        """
        sys.stdout = self._original_stdout 
        pass

class InterceptToolCall(Exception):
    """Custom exception to stop the agent's chain when a tool is intercepted."""
    pass
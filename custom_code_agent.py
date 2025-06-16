import streamlit as st
import os
import sys
from io import StringIO
import base64
from pathlib import Path
import re
import json
import pandas as pd
from datetime import datetime

# Import your agent executor and the specific tool instance
from agent.core import get_agent_executor
from tools.python_executor import PythonCodeExecutorTool
from utils.callbacks import StreamlitCodeExecutionCallbackHandler, InterceptToolCall
from tools.task_completed import TaskCompletedTool

# NEW: Import LangChain message types for injecting history into agent memory
from langchain_core.messages import HumanMessage, AIMessage

# NEW: Import chat utility functions
from utils.chat_utils import generate_chat_name

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# --- Streamlit UI Setup ---
st.set_page_config(page_title="AI Code Assistant", page_icon="🤖")

st.title("AI Code Assistant (Groq)")
st.caption("A ReAct agent powered by Groq (Llama3) for Python code generation and execution.")

# --- Session State Initialization and Reset Function ---
def initialize_session_state():
    """Initializes or resets all relevant session state variables for a new chat."""

    # --- MODIFIED/NEW: Master list for all conversations and current index ---
    # Ensure 'all_conversations' is only initialized once per session (i.e., per 'streamlit run' command)
    if "all_conversations" not in st.session_state:
        st.session_state.all_conversations = [] # List of dictionaries: {'name': str, 'messages': list}
        st.session_state.current_conversation_index = -1 # No chat active initially

    # Ensure there's always at least one conversation to start with on initial app load,
    # or if all previous chats were somehow cleared.
    if not st.session_state.all_conversations:
        st.session_state.all_conversations.append({
            'name': 'New Chat', # Default name for the first chat
            'messages': [{"role": "assistant", "content": "Hello! I am your AI Code Assistant. How can I help you with Python today?"}]
        })
        st.session_state.current_conversation_index = 0
    # This ensures that if the app reruns (e.g., on code change or browser refresh) and
    # the index might have defaulted to -1 due to other logic, it correctly points to an existing chat.
    elif st.session_state.current_conversation_index == -1: 
         st.session_state.current_conversation_index = len(st.session_state.all_conversations) - 1


    # Link st.session_state.messages to the currently active conversation in the list.
    # This is critical: any append/modification to st.session_state.messages
    # will directly affect the chosen conversation in st.session_state.all_conversations.
    st.session_state.messages = st.session_state.all_conversations[st.session_state.current_conversation_index]['messages']
    # --- END MODIFIED/NEW ---

    # Initialize Agent Executor and Tools
    # This block re-initializes the agent (and its memory) ONLY when 'needs_reinitialization' is True.
    # This flag is set when switching chats or starting a new chat, ensuring the agent has correct context.
    if "agent_executor" not in st.session_state or st.session_state.get("needs_reinitialization", False):
        try:
            # Convert current Streamlit messages to LangChain message format for agent memory
            lc_chat_history = []
            for msg in st.session_state.messages:
                if isinstance(msg["content"], str):
                    if msg["role"] == "user":
                        lc_chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        lc_chat_history.append(AIMessage(content=msg["content"]))
                elif isinstance(msg["content"], dict) and msg["content"].get("type") == "code_execution":
                    # For code execution, include both the code input and the output for agent's context
                    lc_chat_history.append(HumanMessage(content=f"Executed Python code:\n```python\n{msg['content']['code']}\n```"))
                    lc_chat_history.append(AIMessage(content=f"Code Execution Result:\n```\n{msg['content']['output']}\n```"))
            
            agent_exec = get_agent_executor()
            # --- THE CRITICAL FIX FOR LANGCHAIN MEMORY ---
            # Clear any existing messages in the buffer (from previous initializations)
            agent_exec.memory.clear()
            # Extend the buffer with the current conversation's history
            agent_exec.memory.buffer.extend(lc_chat_history)
            # --- END OF CRITICAL FIX ---

            python_executor_tool = next(tool for tool in agent_exec.tools if tool.name == "python_code_executor")
            task_completed_tool = next(tool for tool in agent_exec.tools if tool.name == "task_completed")
            callback_logs_buffer = StringIO()
            callback_handler = StreamlitCodeExecutionCallbackHandler(python_executor_tool, task_completed_tool, callback_logs_buffer)
            agent_exec.callbacks = [callback_handler]

            st.session_state.agent_executor = agent_exec
            st.session_state.python_executor_tool = python_executor_tool
            st.session_state.task_completed_tool = task_completed_tool
            st.session_state.callback_handler = callback_handler
            st.session_state.callback_logs_buffer = callback_logs_buffer
            st.session_state.needs_reinitialization = False # Reset flag after successful re-init
        except (ValueError, RuntimeError) as e:
            st.error(f"Failed to initialize Agent: {e}. Please ensure your Groq API key is correct and valid. Check Groq's model deprecation notices.")
            st.stop()

    # State for Human-in-the-Loop flow and Final Answer (keep as is)
    if "pending_action" not in st.session_state:
        st.session_state.pending_action = None
    if "pending_final_answer" not in st.session_state:
        st.session_state.pending_final_answer = None
    if "last_processed_observation" not in st.session_state:
        st.session_state.last_processed_observation = None
    if "agent_continuation_needed" not in st.session_state:
        st.session_state.agent_continuation_needed = False
    if "last_agent_action_log_entry" not in st.session_state:
        st.session_state.last_agent_action_log_entry = None
    if "current_agent_chain_user_prompt" not in st.session_state:
        st.session_state.current_agent_chain_user_prompt = None
    if "last_executed_code" not in st.session_state:
        st.session_state.last_executed_code = None
    if "last_successful_output" not in st.session_state:
        st.session_state.last_successful_output = None
    if "last_generated_chart_data" not in st.session_state:
        st.session_state.last_generated_chart_data = None
    if "last_generated_plot_file" not in st.session_state:
        st.session_state.last_generated_plot_file = None
    if "execution_count" not in st.session_state:  # Track number of code executions in a chain
        st.session_state.execution_count = 0

    # Flags to handle new user prompts vs agent continuations (keep as is)
    if "last_user_prompt" not in st.session_state:
        st.session_state.last_user_prompt = None

initialize_session_state()

# Function to reset the chat for a new conversation
def reset_chat_and_agent():
    """Adds a new empty chat to the list and makes it the active one."""
    # The current conversation is automatically saved as st.session_state.messages is a reference
    
    # Create a new, empty conversation with a default name
    new_chat_entry = {
        'name': 'New Chat',
        'messages': [{"role": "assistant", "content": "Hello! I am your AI Code Assistant. How can I help you with Python today?"}]
    }
    st.session_state.all_conversations.append(new_chat_entry)
    st.session_state.current_conversation_index = len(st.session_state.all_conversations) - 1
    
    # Update st.session_state.messages to point to this new conversation's messages
    st.session_state.messages = st.session_state.all_conversations[st.session_state.current_conversation_index]['messages']

    st.session_state.needs_reinitialization = True # Force agent re-init for new chat
    st.session_state.pending_action = None
    st.session_state.pending_final_answer = None
    st.session_state.agent_continuation_needed = False
    st.session_state.last_processed_observation = None
    st.session_state.last_agent_action_log_entry = None
    st.session_state.current_agent_chain_user_prompt = None
    st.session_state.last_user_prompt = None
    st.session_state.last_executed_code = None
    st.session_state.last_successful_output = None
    st.session_state.last_generated_chart_data = None
    st.session_state.last_generated_plot_file = None
    st.session_state.execution_count = 0
    st.rerun()

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    groq_api_key_env = os.getenv("GROQ_API_KEY")
    if not groq_api_key_env:
        st.warning("GROQ_API_KEY not found in environment variables. Please set it or enter it below.")
        groq_api_key_input = st.text_input("Enter your Groq API Key:", type="password")
        if groq_api_key_input:
            os.environ["GROQ_API_KEY"] = groq_api_key_input
            st.session_state.needs_reinitialization = True
            st.rerun()
    else:
        st.success("Groq API Key detected from environment.")

    st.markdown("---")
    
    # --- MODIFIED: Conversation Selector to display names and handle switching ---
    chat_names = [conv['name'] for conv in st.session_state.all_conversations]
    
    # Ensure index is valid, especially on first load or after adding a new chat
    current_idx_for_selectbox = st.session_state.current_conversation_index 
    
    # Use a unique key for the selectbox to prevent Streamlit errors if options change frequently
    selector_key = f"chat_selector_{len(st.session_state.all_conversations)}" 

    selected_chat_label = st.selectbox(
        "Select Conversation", 
        chat_names, # Use chat_names for display
        index=current_idx_for_selectbox, 
        key=selector_key
    )

    # Logic to switch chat if a different one is selected
    if selected_chat_label and chat_names: # Ensure chat_names is not empty
        new_index = chat_names.index(selected_chat_label)
        if new_index != st.session_state.current_conversation_index:
            # The current conversation's state is implicitly saved because st.session_state.messages
            # is a live reference to the list within st.session_state.all_conversations.
            
            st.session_state.current_conversation_index = new_index
            st.session_state.messages = st.session_state.all_conversations[new_index]['messages'] # Update the messages pointer

            st.session_state.needs_reinitialization = True # Force agent re-initialization with the new chat's history
            
            # Reset any pending HIL states for a clean transition to a different chat context
            st.session_state.pending_action = None
            st.session_state.pending_final_answer = None
            st.session_state.agent_continuation_needed = False
            st.session_state.last_processed_observation = None
            st.session_state.last_agent_action_log_entry = None
            st.session_state.current_agent_chain_user_prompt = None
            st.session_state.last_user_prompt = None
            st.session_state.last_executed_code = None
            st.session_state.last_successful_output = None
            st.session_state.last_generated_chart_data = None
            st.session_state.last_generated_plot_file = None
            st.session_state.execution_count = 0

            st.rerun()
    # --- END MODIFIED ---
    
    if st.button("New Chat", on_click=reset_chat_and_agent, help="Start a fresh conversation and clear agent memory."):
        pass # The on_click handler will trigger the logic and rerun

    st.markdown("---")
    st.markdown("Developed with Streamlit and LangChain.")

# --- Display Chat Messages ---
# This loop iterates through the messages of the *currently active* conversation.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict):
            if message["content"].get("type") == "chart":
                st.subheader(message["content"].get("title", "Generated Chart"))
                try:
                    df = message["content"]["data"]
                    x_label = message["content"].get("x_label")
                    y_label = message["content"].get("y_label")
                    if x_label and y_label and x_label in df.columns and y_label in df.columns:
                        st.line_chart(df, x=x_label, y=y_label, use_container_width=True)
                    else:
                        if len(df.columns) >= 2:
                            x_col, y_col = df.columns[0], df.columns[1]
                            st.line_chart(df, x=x_col, y=y_col, use_container_width=True)
                            st.info(f"Chart labels for x/y axis not explicitly provided. Using '{x_col}' as x-axis and '{y_col}' as y-axis.")
                        else:
                            st.error("Chart data must have at least two columns for plotting.")
                except Exception as chart_error:
                    st.error(f"Error rendering chart: {chart_error}")
                st.caption("Generated visualization. AI models can make mistakes. Always check original sources.")
            
            elif message["content"].get("type") == "file_display":
                if message["content"]["mime"].startswith("image/"):
                    st.image(message["content"]["data"], caption=message["content"].get("caption", "Generated Image"), use_container_width=True)
                else:
                    st.info(f"File created: {message['content']['file_name']}")
                
                if message["content"].get("download_label"):
                    st.download_button(
                        label=message["content"].get("download_label"),
                        data=message["content"].get("data"),
                        file_name=message["content"].get("file_name", "download.bin"),
                        mime=message["content"].get("mime", "application/octet-stream")
                    )
            # --- Code execution messages are now handled here for persistence ---
            elif message["content"].get("type") == "code_execution":
                st.markdown("**Agent proposed code:**")
                st.code(message["content"]["code"], language="python")
                st.markdown("**Code Execution Result:**")
                st.code(message["content"]["output"], language="bash") # Use bash for generic output/error, or 'text'
            
            elif "content_text" in message["content"]:
                st.markdown(message["content"]["content_text"])
        else:
            st.markdown(message["content"])

# --- Human-in-the-Loop (HIL) Section for Code Execution ---
# This section only appears when the agent proposes code and waits for approval.
if st.session_state.pending_action:
    with st.chat_message("assistant"):
        proposed_code = st.session_state.pending_action["tool_input"]

        st.warning("The agent proposes to execute the following Python code:")
        st.code(proposed_code, language="python")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Approve & Execute Code", key="approve_code"):
                with st.spinner("Executing proposed code..."):
                    execution_result = st.session_state.python_executor_tool.execute_code_after_approval(
                        proposed_code
                    )
                
                # --- MODIFIED: Store code and result in a structured message for persistent display ---
                code_execution_message = {
                    "role": "assistant",
                    "content": {
                        "type": "code_execution",
                        "code": proposed_code,
                        "output": execution_result
                    }
                }
                # Append this structured message to the current conversation's messages
                # This ensures the code and output remain visible after execution.
                # The check prevents redundant appending on reruns triggered by other Streamlit events.
                if not (st.session_state.messages and 
                        isinstance(st.session_state.messages[-1]["content"], dict) and 
                        st.session_state.messages[-1]["content"].get("type") == "code_execution" and
                        st.session_state.messages[-1]["content"].get("code") == proposed_code and
                        st.session_state.messages[-1]["content"].get("output") == execution_result):
                    st.session_state.messages.append(code_execution_message)
                else:
                    # If the message is already there (e.g. from an F5 refresh when this was the last action),
                    # just give a small info message to the user for clarity.
                    st.info("Code executed. Result is identical to previous output and already displayed above.")
                # --- END MODIFIED ---
                
                st.session_state.execution_count += 1
                st.session_state.last_executed_code = proposed_code
                st.session_state.last_successful_output = execution_result if "Standard Error:" not in execution_result else None

                # Reset chart/plot file states before new detection
                st.session_state.last_generated_chart_data = None
                st.session_state.last_generated_plot_file = None

                # --- Process Output for PLOT_DATA_JSON_START ---
                # This logic checks the execution_result string for plot data or file creation.
                # It remains largely the same as before.
                plot_data_match = re.search(r"PLOT_DATA_JSON_START:(.*):PLOT_DATA_JSON_END", execution_result, re.DOTALL)
                if plot_data_match:
                    try:
                        plot_data_json = plot_data_match.group(1).strip()
                        parsed_plot_data = json.loads(plot_data_json)
                        # ... (rest of plot data parsing and chart_data population) ...
                        df_chart = pd.DataFrame(parsed_plot_data)
                        chart_title = "Generated Chart" # Simplified title logic for brevity here
                        x_label_key = df_chart.columns[0] if len(df_chart.columns) > 0 else None
                        y_label_key = df_chart.columns[1] if len(df_chart.columns) > 1 else None

                        st.session_state.last_generated_chart_data = {
                            "type": "chart",
                            "data": df_chart,
                            "title": chart_title,
                            "x_label": x_label_key,
                            "y_label": y_label_key
                        }
                        execution_result = execution_result.replace(f"PLOT_DATA_JSON_START:{plot_data_json}:PLOT_DATA_JSON_END", "").strip()

                    except (json.JSONDecodeError, ValueError) as e:
                        st.warning(f"Failed to parse plot data from agent output: {e}. Raw JSON: {plot_data_json}")
                        st.session_state.last_generated_chart_data = None
                else:
                    st.session_state.last_generated_chart_data = None

                # --- Process Output for Generated Plot FILES ---
                # This also remains largely the same.
                file_creation_match = re.search(r"Files created during execution: (.*)", execution_result, re.DOTALL)
                detected_filenames = []
                if file_creation_match:
                    detected_filenames_str = file_creation_match.group(1)
                    detected_filenames = [f.strip() for f in detected_filenames_str.split(',')]
                    execution_result = execution_result.replace(f"Files created during execution: {detected_filenames_str}", "").strip()
                
                for filename in detected_filenames:
                    file_path = Path(filename)
                    if file_path.is_file():
                        mime_type = "application/octet-stream"
                        if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif']:
                            mime_type = f"image/{file_path.suffix.lower().strip('.')}"
                        elif file_path.suffix.lower() == '.csv':
                            mime_type = "text/csv"
                        elif file_path.suffix.lower() == '.txt':
                            mime_type = "text/plain"

                        try:
                            with open(file_path, "rb") as f:
                                file_bytes = f.read()
                            
                            file_display_entry = {
                                "type": "file_display",
                                "data": file_bytes,
                                "caption": f"Generated: {filename}",
                                "download_label": f"Download {filename}",
                                "file_name": filename,
                                "mime": mime_type
                            }
                            if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif']:
                                st.session_state.last_generated_plot_file = file_display_entry
                            else:
                                st.session_state.messages.append({"role": "assistant", "content": file_display_entry})
                                
                        except Exception as e:
                            st.warning(f"Could not load or display file {filename}: {e}")
                            st.session_state.last_generated_plot_file = None
                    else:
                        st.warning(f"Agent mentioned file '{filename}' but it does not exist.")
                
                st.session_state.last_processed_observation = execution_result.strip()
                st.session_state.agent_continuation_needed = True
                
                st.session_state.pending_action = None
                st.rerun()
        with col2:
            if st.button("Cancel Execution", key="cancel_code"):
                cancellation_message = "Code execution CANCELED by user. Agent will proceed with this observation."
                st.session_state.messages.append({"role": "assistant", "content": cancellation_message})
                
                st.session_state.last_processed_observation = cancellation_message
                st.session_state.agent_continuation_needed = True
                
                st.session_state.pending_action = None
                st.rerun()

# --- Handle Task Completed interception ---
# This section executes when the agent signals completion via TaskCompletedTool.
if st.session_state.pending_final_answer:
    with st.chat_message("assistant"):
        st.success("Task Completed!")
        
        # Display charts/files associated with the final answer
        if st.session_state.last_generated_chart_data:
            with st.container():
                st.markdown("---")
                chart_msg_content = st.session_state.last_generated_chart_data
                st.subheader(chart_msg_content.get("title", "Generated Chart"))
                try:
                    df = chart_msg_content["data"]
                    x_label = chart_msg_content.get("x_label")
                    y_label = chart_msg_content.get("y_label")
                    if x_label and y_label and x_label in df.columns and y_label in df.columns:
                        st.line_chart(df, x=x_label, y=y_label, use_container_width=True)
                    else:
                        if len(df.columns) >= 2:
                            x_col, y_col = df.columns[0], df.columns[1]
                            st.line_chart(df, x=x_col, y=y_col, use_container_width=True)
                            st.info(f"Chart labels for x/y axis not explicitly provided. Using '{x_col}' as x-axis and '{y_col}' as y-axis.")
                        else:
                            st.error("Chart data must have at least two columns for plotting.")
                except Exception as chart_error:
                    st.error(f"Error re-rendering chart: {chart_error}")
                st.caption("Generated visualization. AI models can make mistakes. Always check original sources.")
                st.markdown("---")
        
        if st.session_state.last_generated_plot_file:
            with st.container():
                st.markdown("---")
                file_msg_content = st.session_state.last_generated_plot_file
                st.image(file_msg_content["data"], caption=file_msg_content.get("caption", "Generated Plot"), use_container_width=True)
                st.download_button(
                    label=file_msg_content.get("download_label"),
                    data=file_msg_content.get("data"),
                    file_name=file_msg_content.get("file_name", "download.png"),
                    mime=file_msg_content.get("mime", "image/png")
                )
                st.caption("Generated visualization. AI models can make mistakes. Always check original sources.")
                st.markdown("---")

        # Display the final answer text
        if isinstance(st.session_state.pending_final_answer, dict) and "content_text" in st.session_state.pending_final_answer:
            st.markdown(st.session_state.pending_final_answer["content_text"])
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.pending_final_answer["content_text"]})
        else:
            st.markdown(st.session_state.pending_final_answer)
            st.session_state.messages.append({"role": "assistant", "content": st.session_state.pending_final_answer})

        # Optionally show agent's final thought process
        if st.session_state.last_agent_action_log_entry:
            with st.expander("Agent's Thought Process (Final Step)"):
                st.code(st.session_state.last_agent_action_log_entry, language='ansi')

        # Reset states for the next user interaction
        st.session_state.pending_final_answer = None
        st.session_state.agent_continuation_needed = False
        st.session_state.last_user_prompt = None
        st.session_state.current_agent_chain_user_prompt = None
        st.session_state.last_agent_action_log_entry = None
        st.session_state.last_executed_code = None
        st.session_state.last_successful_output = None
        st.session_state.last_generated_chart_data = None
        st.session_state.last_generated_plot_file = None
        st.session_state.execution_count = 0
        st.rerun() # Rerun to clear the pending_final_answer UI elements

# --- Agent Invocation Logic ---
# This block runs the agent if there's a new user prompt or the agent needs to continue a task.
if (st.session_state.agent_continuation_needed or st.session_state.get("last_user_prompt")) and \
   not st.session_state.pending_action and not st.session_state.pending_final_answer:
    
    agent_input = None

    if st.session_state.agent_continuation_needed:
        # Agent is continuing a multi-step task, append observation to scratchpad
        base_scratchpad = st.session_state.last_agent_action_log_entry if st.session_state.last_agent_action_log_entry else ""
        current_observation = st.session_state.last_processed_observation
        agent_input = {
            "input": st.session_state.current_agent_chain_user_prompt,
            "agent_scratchpad": base_scratchpad + f"\nObservation: {current_observation}"
        }
        
        st.session_state.agent_continuation_needed = False # Reset flag for current turn
        st.session_state.last_processed_observation = None
        st.session_state.last_agent_action_log_entry = None
        
        # --- MODIFIED COMPLETION LOGIC (Fallback) ---
        # This auto-completion is a fallback if the agent doesn't explicitly call task_completed.
        # It's here to prevent the UI from getting stuck if the agent concludes without calling the tool.
        if "Standard Output:" in current_observation and "Standard Error:" not in current_observation:
            raw_output = current_observation.replace("Standard Output:", "").strip()
            
            # Auto-complete if it's a short interaction (single execution) or many executions (potential loop)
            if st.session_state.current_agent_chain_user_prompt and (st.session_state.execution_count == 1 or st.session_state.execution_count > 5):
                final_answer_text = f"Task completed based on code execution. The output is: {raw_output}"
                
                if st.session_state.last_generated_chart_data:
                    chart_title_for_final = st.session_state.last_generated_chart_data.get("title", "Generated Chart")
                    final_answer_text += f"\n\nA chart titled '{chart_title_for_final}' has been displayed above to visualize the data."
                elif st.session_state.last_generated_plot_file:
                    file_name_for_final = st.session_state.last_generated_plot_file.get("file_name", "a generated plot file")
                    final_answer_text += f"\n\nA plot saved as '{file_name_for_final}' has been displayed above."
                
                st.session_state.pending_final_answer = {"type": "final_answer_auto_completed", "content_text": final_answer_text}
                st.rerun() # Trigger a rerun to display the final answer
        
    elif st.session_state.get("last_user_prompt"):
        # New user prompt, start a new agent invocation chain
        agent_input = {"input": st.session_state.last_user_prompt}
        st.session_state.current_agent_chain_user_prompt = st.session_state.last_user_prompt # Store original prompt for chain
        st.session_state.last_user_prompt = None # Clear for next turn
        st.session_state.execution_count = 0 # Reset execution counter for new task

    if agent_input: # Only invoke if there's valid input for the agent
        with st.chat_message("assistant"):
            thought_process_placeholder = st.empty() # Placeholder for agent's thinking logs

            try:
                st.session_state.callback_logs_buffer.seek(0) # Clear previous logs
                st.session_state.callback_logs_buffer.truncate(0)

                with st.spinner("Agent is thinking..."):
                    response = st.session_state.agent_executor.invoke(agent_input)
                    agent_output = response["output"]

                captured_logs = st.session_state.callback_logs_buffer.getvalue()

                with thought_process_placeholder.expander("Agent's Thought Process"):
                    st.code(captured_logs, language='ansi') # Display agent's raw thoughts

                st.markdown(f"**Agent's Final Response:** {agent_output}") # Display final human-readable output from agent
                st.session_state.messages.append({"role": "assistant", "content": agent_output}) # Add to chat history


                # --- Process any files mentioned in agent_output for display/download ---
                file_matches = re.findall(r"Files created during execution: (.*)", agent_output, re.DOTALL)
                
                if file_matches:
                    detected_filenames_str = file_matches[0]
                    detected_filenames = [f.strip() for f in detected_filenames_str.split(',')]
                    
                    for filename in detected_filenames:
                        file_path = Path(filename)
                        if file_path.is_file():
                            try:
                                with open(file_path, "rb") as f:
                                    file_bytes = f.read()
                                
                                mime_type = "application/octet-stream" # Default MIME type
                                if file_path.suffix.lower() == ".gif":
                                    mime_type = "image/gif"
                                elif file_path.suffix.lower() == ".png":
                                    mime_type = "image/png"
                                elif file_path.suffix.lower() == ".jpg" or file_path.suffix.lower() == ".jpeg":
                                    mime_type = "image/jpeg"
                                elif file_path.suffix.lower() == ".csv":
                                    mime_type = "text/csv"
                                elif file_path.suffix.lower() == ".txt":
                                    mime_type = "text/plain"
                                
                                file_display_data = {
                                    "type": "file_display",
                                    "data": file_bytes,
                                    "caption": f"Generated: {filename}",
                                    "download_label": f"Download {filename}",
                                    "file_name": filename,
                                    "mime": mime_type
                                }
                                # Store for potential re-display with final answer, or display immediately
                                if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif']:
                                    st.session_state.last_generated_plot_file = file_display_data
                                else:
                                    st.session_state.messages.append({"role": "assistant", "content": file_display_data})
                                
                            except Exception as file_display_e:
                                st.warning(f"Could not load or display file {filename}: {file_display_e}")
                        else:
                            st.warning(f"Agent mentioned file '{filename}' but it does not exist.")
                
                # Reset flags after successful agent turn
                st.session_state.last_user_prompt = None
                st.session_state.agent_continuation_needed = False
                st.session_state.current_agent_chain_user_prompt = None
                st.session_state.last_agent_action_log_entry = None

            except InterceptToolCall:
                pass # This is expected when agent proposes code or task completed
            except Exception as e:
                # --- General Error Handling for Agent Execution ---
                if sys.stdout != st.session_state.callback_handler._original_stdout:
                    sys.stdout = st.session_state.callback_handler._original_stdout # Ensure stdout is restored

                captured_logs = st.session_state.callback_logs_buffer.getvalue()

                error_message_str = str(e)
                display_error_message = f"An unexpected error occurred during agent execution: {error_message_str}"
                
                # Provide more user-friendly messages for common errors
                if "ConnectionError" in error_message_str or "ConnectError" in error_message_str:
                    display_error_message = "Connection Error: Could not reach the Groq API. Please check your internet connection and Groq's service status."
                elif "401" in error_message_str or "Unauthorized" in error_message_str:
                    display_error_message = "Authentication Error: Invalid Groq API Key. Please check your GROQ_API_KEY in the .env file or sidebar."
                elif "413" in error_message_str or "Request too large" in error_message_str:
                    display_error_message = "Token Limit Error: The request to Groq was too large. This can happen with very long conversations. Try a shorter prompt or reset the session."
                elif "rate_limit_exceeded" in error_message_str:
                    display_error_message = "Rate Limit Exceeded: You've sent too many requests to Groq too quickly. Please wait a moment and try again."
                elif "Timeout" in error_message_str:
                    display_error_message = "Timeout Error: The Groq API did not respond in time. This could be due to network issues or high load on Groq's servers."
                elif "pydantic_core.core_schema.ValidationException" in error_message_str:
                    display_error_message = f"Agent Parsing Error: The agent's output did not conform to the expected format. This can sometimes be resolved by regenerating the response. Details: {e}"
                elif "langchain_core.exceptions.OutputParserException" in error_message_str:
                    display_error_message = f"Agent Output Parsing Error: The agent's output could not be parsed according to ReAct format. This indicates the LLM deviated from the prompt. Try asking the question again or starting a new chat. Details: {e}"
                elif "Input to PromptTemplate is missing variables" in error_message_str: # Catch the specific prompt error
                    display_error_message = f"Prompt Template Error: The agent's output included unescaped curly braces in its internal thought. This is an internal issue; please try again or reset the chat. Details: {e}"


                st.error(display_error_message) # Display error to user

                with thought_process_placeholder.expander("Agent's Process before error"):
                    st.code(captured_logs, language='ansi') # Show agent's thought leading to error
                
                # Log the error message in the chat history
                st.session_state.messages.append({"role": "assistant", "content": f"**Error:** {display_error_message}\n\n**Agent's Process:**\n```ansi\n{captured_logs}\n```"})
                
                # Reset agent state after an error to avoid getting stuck
                st.session_state.needs_reinitialization = True
                st.session_state.pending_action = None
                st.session_state.pending_final_answer = None
                st.session_state.agent_continuation_needed = False
                st.session_state.last_user_prompt = None
                st.session_state.current_agent_chain_user_prompt = None
                st.session_state.last_agent_action_log_entry = None
                st.session_state.execution_count = 0
                
                st.session_state.messages.append({"role": "assistant", "content": "*(Agent session reset due to error. Please start a new conversation using 'New Chat' button.)*"})
                st.rerun() # Rerun to reflect the error state and reset button

# --- Chat Input ---
# This input field is displayed only when the agent is not busy and no HIL is pending.
if not st.session_state.pending_action and \
   not st.session_state.agent_continuation_needed and \
   not st.session_state.pending_final_answer and \
   not st.session_state.get("needs_reinitialization", False):
    
    user_prompt = st.chat_input("Ask me to write or execute some Python code...")
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        st.session_state.last_user_prompt = user_prompt
        # --- NEW: Update chat name on first user prompt in a 'New Chat' ---
        # Get the current chat entry from the all_conversations list
        current_chat_entry = st.session_state.all_conversations[st.session_state.current_conversation_index]
        # Only update the name if it's still the default "New Chat" name
        if current_chat_entry['name'] == 'New Chat':
            current_chat_entry['name'] = generate_chat_name(user_prompt)
        # --- END NEW ---
        st.rerun() # Trigger a rerun to process the new user prompt
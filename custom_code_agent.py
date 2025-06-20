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
    # ## UI UPDATE ##: Main state for conversation management
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "current_conversation_id" not in st.session_state:
        # Create the very first conversation
        first_conv_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.conversations[first_conv_id] = {
            "messages": [{"role": "assistant", "content": "Hello! I am your AI Code Assistant. How can I help you with Python today?"}],
            "display_name": "New Conversation" # A temporary display name
        }
        st.session_state.current_conversation_id = first_conv_id

    # Initialize Agent Executor and Tools
    if "agent_executor" not in st.session_state or st.session_state.get("needs_reinitialization", False):
        try:
            agent_exec = get_agent_executor()
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
            st.session_state.needs_reinitialization = False
        except (ValueError, RuntimeError) as e:
            st.error(f"Failed to initialize Agent: {e}. Please ensure your Groq API key is correct and valid. Check Groq's model deprecation notices.")
            st.stop()

    # State for Human-in-the-Loop flow and Final Answer
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
    if "execution_count" not in st.session_state:
        st.session_state.execution_count = 0

    # Flags to handle new user prompts vs agent continuations
    if "last_user_prompt" not in st.session_state:
        st.session_state.last_user_prompt = None

initialize_session_state()

# ## UI UPDATE ##: Helper to get the messages of the current conversation
def get_current_messages():
    conv_id = st.session_state.current_conversation_id
    return st.session_state.conversations[conv_id]["messages"]

# Function to reset the chat for a new conversation
def start_new_chat():
    """Creates a new, separate conversation."""
    # Reset agent state variables
    st.session_state.needs_reinitialization = True
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

    # ## UI UPDATE ##: Create a new conversation entry instead of clearing old one
    new_conv_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.conversations[new_conv_id] = {
        "messages": [{"role": "assistant", "content": "Hello! I am your AI Code Assistant. How can I help you with Python today?"}],
        "display_name": "New Conversation"
    }
    st.session_state.current_conversation_id = new_conv_id
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
    if st.button("New Chat", on_click=start_new_chat, help="Start a fresh conversation and clear agent memory."):
        pass

    # ## UI UPDATE ##: Conversation Selector
    st.markdown("---")
    st.header("Conversations")
    # Create a list of display names for the dropdown
    display_names = [conv["display_name"] for conv in st.session_state.conversations.values()]
    conv_ids = list(st.session_state.conversations.keys())

    # Find the index of the current conversation to set the default for the selectbox
    try:
        current_index = conv_ids.index(st.session_state.current_conversation_id)
    except ValueError:
        current_index = 0

    selected_display_name = st.selectbox(
        "Select Conversation",
        options=display_names,
        index=current_index,
        label_visibility="collapsed"
    )

    # If the selection changed, update the current conversation ID
    selected_id = conv_ids[display_names.index(selected_display_name)]
    if st.session_state.current_conversation_id != selected_id:
        st.session_state.current_conversation_id = selected_id
        st.rerun()

    st.markdown("---")
    st.markdown("Developed with Streamlit and LangChain.")

# --- Display Chat Messages ---
for message in get_current_messages():
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
                        # Dynamically select x and y columns if not specified
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
            elif "content_text" in message["content"]:
                st.markdown(message["content"]["content_text"])
        else:
            st.markdown(message["content"])

# --- Human-in-the-Loop (HIL) Section for Code Execution ---
if st.session_state.pending_action:
    with st.chat_message("assistant"):
        proposed_code = st.session_state.pending_action["tool_input"]

        st.warning("The agent proposes to execute the following Python code:")
        st.code(proposed_code, language="python")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Approve & Execute Code", key="approve_code"):
                # ## UI UPDATE ##: Add the proposed code to history for persistence
                code_to_persist = f"**Proposed Code to Execute:**\n```python\n{proposed_code}\n```"
                get_current_messages().append({"role": "assistant", "content": code_to_persist})

                with st.spinner("Executing proposed code..."):
                    execution_result = st.session_state.python_executor_tool.execute_code_after_approval(
                        proposed_code
                    )

                result_display_content = f"**Code Execution Result:**\n```\n{execution_result}\n```"
                get_current_messages().append({"role": "assistant", "content": result_display_content})

                st.session_state.execution_count += 1
                st.session_state.last_executed_code = proposed_code
                st.session_state.last_successful_output = execution_result if "Standard Error:" not in execution_result else None

                # Reset chart/plot file states before new detection
                st.session_state.last_generated_chart_data = None
                st.session_state.last_generated_plot_file = None

                # --- Process Output for PLOT_DATA_JSON_START ---
                plot_data_match = re.search(r"PLOT_DATA_JSON_START:(.*):PLOT_DATA_JSON_END", execution_result, re.DOTALL)
                if plot_data_match:
                    try:
                        plot_data_json = plot_data_match.group(1).strip()
                        parsed_plot_data = json.loads(plot_data_json)

                        # Validate JSON structure
                        if not isinstance(parsed_plot_data, dict) or len(parsed_plot_data) < 2:
                            if not (isinstance(parsed_plot_data, dict) and all(isinstance(v, list) for v in parsed_plot_data.values())):
                                raise ValueError("Plot data must be a dictionary with at least two keys (columns) for line chart or valid list structure.")

                        df_chart = pd.DataFrame(parsed_plot_data)

                        chart_title = "Generated Chart"
                        if st.session_state.last_agent_action_log_entry:
                            thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:)", st.session_state.last_agent_action_log_entry, re.DOTALL)
                            if thought_match:
                                thought_text = thought_match.group(1).strip()
                                thought_title_match = re.search(r"(?:plot|chart|visualize|growth of|trend of)\s+(.*?)(?:\.|,|\n|$)", thought_text, re.IGNORECASE)
                                if thought_title_match:
                                    extracted_title = thought_title_match.group(1).strip()
                                    chart_title = extracted_title.capitalize() if extracted_title else chart_title
                                    chart_title = re.sub(r"^(?:the\s)?(.*?)(?:\s+using.*|\s+for.*|\s+and.*|\s+of.*)?$", r"\1", chart_title, flags=re.IGNORECASE).strip()

                        x_label_key, y_label_key = (None, None)
                        if len(df_chart.columns) >= 2:
                            x_label_key, y_label_key = df_chart.columns[0], df_chart.columns[1]

                        st.session_state.last_generated_chart_data = {
                            "type": "chart", "data": df_chart, "title": chart_title,
                            "x_label": x_label_key, "y_label": y_label_key
                        }
                        execution_result = execution_result.replace(f"PLOT_DATA_JSON_START:{plot_data_json}:PLOT_DATA_JSON_END", "").strip()

                    except (json.JSONDecodeError, ValueError) as e:
                        st.warning(f"Failed to parse plot data from agent output: {e}. Raw JSON: {plot_data_json}")
                        st.session_state.last_generated_chart_data = None
                else:
                    st.session_state.last_generated_chart_data = None

                # --- Process Output for Generated Plot FILES ---
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
                        elif file_path.suffix.lower() == '.csv': mime_type = "text/csv"
                        elif file_path.suffix.lower() == '.txt': mime_type = "text/plain"

                        try:
                            with open(file_path, "rb") as f: file_bytes = f.read()
                            file_display_entry = {
                                "type": "file_display", "data": file_bytes, "caption": f"Generated: {filename}",
                                "download_label": f"Download {filename}", "file_name": filename, "mime": mime_type
                            }
                            if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif']:
                                st.session_state.last_generated_plot_file = file_display_entry
                            else:
                                get_current_messages().append({"role": "assistant", "content": file_display_entry})
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
                get_current_messages().append({"role": "assistant", "content": cancellation_message})
                st.session_state.last_processed_observation = cancellation_message
                st.session_state.agent_continuation_needed = True
                st.session_state.pending_action = None
                st.rerun()

# --- Handle Task Completed interception ---
if st.session_state.pending_final_answer:
    with st.chat_message("assistant"):
        st.success("Task Completed!")
        
        if st.session_state.last_generated_chart_data:
            with st.container():
                st.markdown("---")
                chart_msg_content = st.session_state.last_generated_chart_data
                st.subheader(chart_msg_content.get("title", "Generated Chart"))
                try:
                    df, x_label, y_label = chart_msg_content["data"], chart_msg_content.get("x_label"), chart_msg_content.get("y_label")
                    if x_label and y_label and x_label in df.columns and y_label in df.columns:
                        st.line_chart(df, x=x_label, y=y_label, use_container_width=True)
                    else:
                        if len(df.columns) >= 2:
                            x_col, y_col = df.columns[0], df.columns[1]
                            st.line_chart(df, x=x_col, y=y_col, use_container_width=True)
                            st.info(f"Chart labels for x/y axis not explicitly provided. Using '{x_col}' as x-axis and '{y_col}' as y-axis.")
                        else: st.error("Chart data must have at least two columns for plotting.")
                except Exception as chart_error: st.error(f"Error re-rendering chart: {chart_error}")
                st.caption("Generated visualization. AI models can make mistakes. Always check original sources.")
                st.markdown("---")
        
        if st.session_state.last_generated_plot_file:
            with st.container():
                st.markdown("---")
                file_msg_content = st.session_state.last_generated_plot_file
                st.image(file_msg_content["data"], caption=file_msg_content.get("caption", "Generated Plot"), use_container_width=True)
                st.download_button(
                    label=file_msg_content.get("download_label"), data=file_msg_content.get("data"),
                    file_name=file_msg_content.get("file_name", "download.png"), mime=file_msg_content.get("mime", "image/png")
                )
                st.caption("Generated visualization. AI models can make mistakes. Always check original sources.")
                st.markdown("---")

        if isinstance(st.session_state.pending_final_answer, dict) and "content_text" in st.session_state.pending_final_answer:
            st.markdown(st.session_state.pending_final_answer["content_text"])
            get_current_messages().append({"role": "assistant", "content": st.session_state.pending_final_answer["content_text"]})
        else:
            st.markdown(st.session_state.pending_final_answer)
            get_current_messages().append({"role": "assistant", "content": st.session_state.pending_final_answer})

        if st.session_state.last_agent_action_log_entry:
            with st.expander("Agent's Thought Process (Final Step)"):
                st.code(st.session_state.last_agent_action_log_entry, language='ansi')

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

# --- Agent Invocation Logic ---
if (st.session_state.agent_continuation_needed or st.session_state.get("last_user_prompt")) and \
   not st.session_state.pending_action and not st.session_state.pending_final_answer:
    
    agent_input = None

    if st.session_state.agent_continuation_needed:
        base_scratchpad = st.session_state.last_agent_action_log_entry if st.session_state.last_agent_action_log_entry else ""
        current_observation = st.session_state.last_processed_observation
        agent_input = {
            "input": st.session_state.current_agent_chain_user_prompt,
            "agent_scratchpad": base_scratchpad + f"\nObservation: {current_observation}"
        }
        
        st.session_state.agent_continuation_needed = False
        st.session_state.last_processed_observation = None
        st.session_state.last_agent_action_log_entry = None
        
        if "Standard Output:" in current_observation and "Standard Error:" not in current_observation:
            raw_output = current_observation.replace("Standard Output:", "").strip()
            
            if st.session_state.current_agent_chain_user_prompt and (st.session_state.execution_count == 1 or st.session_state.execution_count > 5):
                final_answer_text = f"Task completed based on code execution. The output is: {raw_output}"
                
                if st.session_state.last_generated_chart_data:
                    chart_title_for_final = st.session_state.last_generated_chart_data.get("title", "Generated Chart")
                    final_answer_text += f"\n\nA chart titled '{chart_title_for_final}' has been displayed above to visualize the data."
                elif st.session_state.last_generated_plot_file:
                    file_name_for_final = st.session_state.last_generated_plot_file.get("file_name", "a generated plot file")
                    final_answer_text += f"\n\nA plot saved as '{file_name_for_final}' has been displayed above."
                
                st.session_state.pending_final_answer = {"type": "final_answer_auto_completed", "content_text": final_answer_text}
                st.rerun()
        
    elif st.session_state.get("last_user_prompt"):
        agent_input = {"input": st.session_state.last_user_prompt}
        st.session_state.current_agent_chain_user_prompt = st.session_state.last_user_prompt
        st.session_state.last_user_prompt = None
        st.session_state.execution_count = 0

    if agent_input:
        with st.chat_message("assistant"):
            thought_process_placeholder = st.empty()
            try:
                st.session_state.callback_logs_buffer.seek(0)
                st.session_state.callback_logs_buffer.truncate(0)

                with st.spinner("Agent is thinking..."):
                    response = st.session_state.agent_executor.invoke(agent_input)
                    agent_output = response["output"]

                captured_logs = st.session_state.callback_logs_buffer.getvalue()
                with thought_process_placeholder.expander("Agent's Thought Process"):
                    st.code(captured_logs, language='ansi')

                st.markdown(f"**Agent's Final Response:** {agent_output}")
                get_current_messages().append({"role": "assistant", "content": agent_output})

                file_matches = re.findall(r"Files created during execution: (.*)", agent_output, re.DOTALL)
                if file_matches:
                    detected_filenames_str = file_matches[0]
                    detected_filenames = [f.strip() for f in detected_filenames_str.split(',')]
                    
                    for filename in detected_filenames:
                        file_path = Path(filename)
                        if file_path.is_file():
                            try:
                                with open(file_path, "rb") as f: file_bytes = f.read()
                                mime_type = "application/octet-stream" # Default
                                if file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif"]: mime_type = f"image/{file_path.suffix.lower().lstrip('.')}"
                                elif file_path.suffix.lower() == ".csv": mime_type = "text/csv"
                                elif file_path.suffix.lower() == ".txt": mime_type = "text/plain"
                                
                                file_display_data = {
                                    "type": "file_display", "data": file_bytes, "caption": f"Generated: {filename}",
                                    "download_label": f"Download {filename}", "file_name": filename, "mime": mime_type
                                }
                                get_current_messages().append({"role": "assistant", "content": file_display_data})
                            except Exception as file_display_e: st.warning(f"Could not display or prepare download for generated file ({filename}): {file_display_e}")
                        else: st.warning(f"Agent mentioned file '{filename}' but it does not exist.")
                
                st.session_state.last_user_prompt = None
                st.session_state.agent_continuation_needed = False
                st.session_state.current_agent_chain_user_prompt = None
                st.session_state.last_agent_action_log_entry = None

            except InterceptToolCall: pass
            except Exception as e:
                if sys.stdout != st.session_state.callback_handler._original_stdout:
                    sys.stdout = st.session_state.callback_handler._original_stdout

                captured_logs = st.session_state.callback_logs_buffer.getvalue()
                error_message_str = str(e)
                # ... (Error handling logic is unchanged)
                display_error_message = f"An unexpected error occurred during agent execution: {error_message_str}"
                st.error(display_error_message)

                with thought_process_placeholder.expander("Agent's Process before error"):
                    st.code(captured_logs, language='ansi')
                
                get_current_messages().append({"role": "assistant", "content": f"**Error:** {display_error_message}\n\n**Agent's Process:**\n```ansi\n{captured_logs}\n```"})
                st.session_state.needs_reinitialization = True
                st.session_state.pending_action = None
                st.session_state.pending_final_answer = None
                st.session_state.agent_continuation_needed = False
                st.session_state.last_user_prompt = None
                st.session_state.current_agent_chain_user_prompt = None
                st.session_state.last_agent_action_log_entry = None
                st.session_state.execution_count = 0
                get_current_messages().append({"role": "assistant", "content": "*(Agent session reset due to error. Please start a new conversation using 'New Chat' button.)*"})
                st.rerun()

# --- Chat Input ---
if not st.session_state.pending_action and \
   not st.session_state.agent_continuation_needed and \
   not st.session_state.pending_final_answer and \
   not st.session_state.get("needs_reinitialization", False):
    
    user_prompt = st.chat_input("Ask me to write or execute some Python code...")
    if user_prompt:
        current_conv_id = st.session_state.current_conversation_id
        current_conv = st.session_state.conversations[current_conv_id]
        
        # ## UI UPDATE ##: Dynamic conversation naming
        # If this is the first user prompt in a new chat, rename the conversation
        if len(current_conv["messages"]) == 1:
            # Truncate long prompts for display name
            new_display_name = user_prompt[:50] + "..." if len(user_prompt) > 50 else user_prompt
            current_conv["display_name"] = new_display_name

        get_current_messages().append({"role": "user", "content": user_prompt})
        st.session_state.last_user_prompt = user_prompt
        st.rerun()
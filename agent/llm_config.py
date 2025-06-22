# agent/llm_config.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import streamlit as st # Import streamlit to access session_state

# Load environment variables from .env file
load_dotenv()

def get_groq_llm():
    """
    Initializes and returns the Groq LLM based on environment variables
    and Streamlit session state.

    Reads configuration for the API key from environment variables.
    Reads model name and temperature from Streamlit's session state.

    Raises:
        ValueError: If GROQ_API_KEY is not found in the environment.
        RuntimeError: If there is an error initializing the LLM.

    Returns:
        ChatGroq: An instance of the configured Groq LLM.
    """
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file or Streamlit input.")

    # Get model parameters from Streamlit session state (preferred for user config)
    # Fallback to environment variables if not in session state (e.g., first run)
    model_name = st.session_state.get("llm_model_name", os.getenv("AGENT_MODEL_NAME", "llama3-70b-8192"))
    temperature = st.session_state.get("llm_temperature", float(os.getenv("AGENT_TEMPERATURE", "0.05")))
    
    # Ensure temperature is float (it already is from slider, but good practice)
    try:
        temperature = float(temperature)
    except (ValueError, TypeError):
        temperature = 0.05
        print(f"Warning: Invalid temperature value. Using default value: {temperature}")


    try:
        llm = ChatGroq(
            temperature=temperature,
            model_name=model_name,
            groq_api_key=GROQ_API_KEY
        )
        return llm
    except Exception as e:
        error_message = (
            f"Error initializing or testing Groq LLM: {e}. "
            f"Please ensure your GROQ_API_KEY is correct and the model name ('{model_name}') is valid. "
            "Check available models at https://console.groq.com/docs/models"
        )
        raise RuntimeError(error_message) from e
# agent/llm_config.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

def get_groq_llm():
    """
    Initializes and returns the Groq LLM based on environment variables.

    Reads configuration for the API key, model name, temperature, and
    max iterations from environment variables, with sensible defaults.

    Raises:
        ValueError: If GROQ_API_KEY is not found in the environment.
        RuntimeError: If there is an error initializing the LLM.

    Returns:
        ChatGroq: An instance of the configured Groq LLM.
    """
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

    # Get model parameters from environment or use defaults
    model_name = os.getenv("AGENT_MODEL_NAME", "llama3-70b-8192")
    try:
        temperature = float(os.getenv("AGENT_TEMPERATURE", "0.05"))
    except (ValueError, TypeError):
        temperature = 0.05
        print(f"Warning: Invalid AGENT_TEMPERATURE. Using default value: {temperature}")


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
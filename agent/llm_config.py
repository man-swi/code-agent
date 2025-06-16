import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv() # Load environment variables here as well, in case this module is imported directly

def get_groq_llm():
    """Initializes and returns the Groq LLM."""
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it or enter it in the Streamlit sidebar.")

    try:
        llm = ChatGroq(
            temperature=0.05,
            model_name="llama3-70b-8192", 
            groq_api_key=GROQ_API_KEY
        )
        return llm
    except Exception as e:
        raise RuntimeError(f"Error initializing or testing Groq LLM: {e}. Please ensure your GROQ_API_KEY is correct and the model name is valid. Check available models at https://console.groq.com/docs/models") from e
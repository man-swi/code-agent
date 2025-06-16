import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

try:
    llm = ChatGroq(
        temperature=0.05, 
        model_name="llama3-8b-8192",
        groq_api_key=GROQ_API_KEY
    )
    response = llm.invoke("What is the capital of France? Respond with only the name of the city.")
    print(f"Direct Groq Test Result: {response.content}")
except Exception as e:
    print(f"Direct Groq Test Failed: {e}")
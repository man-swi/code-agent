# 🤖 AI Code Assistant Agent: A Groq-Powered Python Development Partner

An interactive, intelligent assistant for **Python code generation, safe execution, performance tracking, and dynamic visualization**. Built using **LangChain** and **Groq's Llama3 models**, the application ensures human-in-the-loop safety, clear reasoning, and a seamless coding experience through a sleek **Streamlit interface**.

---

## 🎯 Problem Statement

While many AI assistants generate code, they often lack **explainability, execution control, and performance transparency**. This project bridges those gaps by building a reliable and safe AI agent that:

1. Generates accurate Python code for diverse tasks.
2. Explains its reasoning and planning steps.
3. Awaits **explicit user approval** before executing code (Human-in-the-Loop).
4. Reports execution metrics (time, success/failure).
5. Handles errors gracefully and transparently.
6. Supports file creation and visual output directly from code.

---

## ✨ Key Features

- **🧠 Code Generation**: Understands complex, multi-step requests and writes executable Python code.
- **⚡ Groq-Powered LLM**: Uses `llama3-70b-8192` and `llama3-8b-8192` for intelligent, fast, and high-quality responses.
- **👀 Human-in-the-Loop Execution**:
  - Agent explains the **thought process** and **execution plan**.
  - **Explicit approval** (`✅ Approve & Execute`) required before running any code.
  - **Validates and cleans** code before execution.
- **📊 Performance Metrics**:
  - Tracks execution time.
  - Reports status (`Success`, `Timeout`, `Error`).
  - Logs and displays agent’s reasoning and tool usage.
- **📈 Visualization Support**:
  - Auto-generates charts (line, bar) from JSON outputs.
  - Supports complex visualizations via Matplotlib/Seaborn.
- **📂 File Handling**: Captures and exposes generated files for download.
- **🚨 Robust Error Handling**: Detects and explains errors using logs and feedback.
- **🧩 Streamlit UI**:
  - Sidebar configuration (LLM model, temperature).
  - Dynamic chat history and execution logs.
  - Feedback buttons (`👍` / `👎`) after each interaction.

---

## 🚀 Tech Stack

- **Language**: Python 3.9+
- **Frameworks**:
  - [LangChain](https://www.langchain.com/) (ReAct agent, tool integration)
  - [Streamlit](https://streamlit.io/) (UI and interactivity)
- **LLM Provider**: [Groq](https://console.groq.com/)
- **Models Used**: Llama3 70B and 8B
- **Libraries**:
  - `langchain-core`, `langchain-groq`
  - `pandas`, `matplotlib`, `seaborn`
  - `subprocess`, `sys`, `os`, `re`, `datetime`
  - `python-dotenv`, `pydantic`, `pathlib`

---

## 🏗️ Project Structure

```

.
├── agent/
│   ├── **init**.py
│   ├── core.py              # Agent logic, tools, and execution flow
│   ├── llm\_config.py        # Groq LLM setup
│   ├── prompt.py            # ReAct-style agent instructions
├── tools/
│   ├── **init**.py
│   ├── python\_executor.py   # Code execution + metrics + approval
│   ├── task\_completed.py    # Signals task completion, feedback
├── utils/
│   ├── **init**.py
│   ├── callbacks.py         # Custom LangChain callback for HIL and logging
├── .env                     # GROQ\_API\_KEY stored here
├── custom\_code\_agent.py     # Main Streamlit app
├── requirements.txt         # Dependencies
├── README.md

````

---

## 👤 Contributor

- **Manaswi Kamble**

---

## 🧪 Getting Started

### ✅ Prerequisites

1. **Python**: Install Python 3.9 or newer.
2. **Groq API Key**: [Get your free API key](https://console.groq.com/).

### 🔧 Setup Instructions

```bash
# Clone the repo
git clone <your_repo_url>
cd <your_project_directory>

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # For Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````

### 🔑 Configure Your Groq API Key

* Create a `.env` file in the root:

```dotenv
GROQ_API_KEY="your_groq_api_key_here"
```

OR

* Enter it manually when prompted in the sidebar UI.

---

### 🚀 Run the App

```bash
streamlit run custom_code_agent.py
```

> The app will launch in your browser. Select your LLM model, set temperature, and start coding with your AI partner!

---

## 🌐 Live Demo

🔗 [Try it out on Streamlit](https://code-agent.streamlit.app/)

---

## 📚 Example Interaction

**User:**

> Create a Python function to calculate the factorial of 5.

**AI Assistant:**

> 🧠 *Agent's Thought:*
> I’ll write a recursive factorial function, call it with 5, and print the result.

> 💡 *Generated Code (after approval):*

```python
def calculate_factorial(n):
    if n < 0:
        return "Factorial is not defined for negative numbers."
    elif n == 0:
        return 1
    else:
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

number = 5
factorial_result = calculate_factorial(number)
print(f"The factorial of {number} is: {factorial_result}")
```

> ✅ *Execution Output:*
> `The factorial of 5 is: 120`

---

## 💡 Learnings

* **Prompt Engineering**: Crafting structured ReAct prompts improves reasoning and tool use.
* **Human-in-the-Loop**: Managing Streamlit state to pause execution until user approval.
* **Sandboxed Execution**: Secure use of `subprocess.run()` and careful input sanitization.
* **LangChain Callbacks**: Used `BaseCallbackHandler` to capture and visualize agent steps.
* **UI State Management**: Leveraged `st.session_state` for dynamic flow control.

---

## 🔮 Future Enhancements

* 🧑‍🔧 Code Debugger: Suggest fixes based on traceback analysis.
* 📊 Plotly/Altair Support: Enable interactive charting.
* 🗂 File Awareness: Let agents read from local user files.
* 💾 Conversation Persistence: Save chat history between sessions.
* 🔐 Code Risk Alerts: Warn users about risky patterns (e.g., network access).

---

## 🤝 Feedback & Contributions

We welcome your feedback! Use the `👍` or `👎` buttons after agent responses to help improve this assistant.

---

## 📌 Why This Project Stands Out for AI Roles

* **AI Agent Engineering**: Built a LangChain ReAct agent with advanced capabilities.
* **LLM Deployment Expertise**: Hands-on with Groq API, prompt optimization, and execution pipelines.
* **Safe Code Generation**: Human-in-the-Loop, execution metrics, and explainability.
* **UI/UX**: User-centric Streamlit interface with logging, feedback, and control.

---

> ⚡ Ready to code smarter, faster, and safer? Run the AI Code Assistant and supercharge your Python development workflow!

```

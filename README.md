# AI Code Assistant Agent

Welcome to the **AI Code Assistant Agent**, a powerful Streamlit-based application powered by Groq's Llama3 model. This ReAct agent assists users in generating, executing, and visualizing Python code through an interactive Human-in-the-Loop (HIL) workflow. Designed for developers, educators, and data enthusiasts, it supports a wide range of tasks from simple calculations to complex data visualizations.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Starting the Application](#starting-the-application)
  - [Examples](#examples)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features
- **Interactive Code Generation**: Proposes Python code for user approval before execution.
- **Human-in-the-Loop (HIL)**: Allows manual review and editing of code via the UI.
- **Visualization Support**: Generates line charts (via JSON) and saves complex plots (e.g., PNG files) using Matplotlib/Seaborn.
- **Conversation Management**: Supports multiple chat sessions with dynamic naming.
- **Performance Metrics**: Displays processing time for each task.
- **Feedback System**: Collects user feedback with 👍/👎 options.
- **File Uploads**: Processes uploaded CSV, TXT, and other data files.

## Prerequisites
Before running the application, ensure your system meets the following requirements:
- **Operating System**: Windows, macOS, or Linux (tested on Windows 11).
- **Python**: Version 3.11 or higher.
- **Dependencies**:
  - `streamlit`
  - `langchain`
  - `langchain-groq`
  - `python-dotenv`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `psutil` (for advanced metrics)

## Installation
Follow these steps to set up the AI Code Assistant locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/ai-code-assistant.git
   cd ai-code-assistant
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note*: Create a `requirements.txt` file with the listed dependencies if not already present.

4. **Set Up Environment Variables**:
   - Create a `.env` file in the project root.
   - Add your Groq API key:
     ```
     GROQ_API_KEY=your_api_key_here
     ```
   - Obtain an API key from [Groq Console](https://console.groq.com/).

5. **Verify Installation**:
   - Ensure all dependencies are installed without errors.
   - Check Python version with `python --version`.

## Usage

### Starting the Application
1. Activate the virtual environment (if used):
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Run the application:
   ```bash
   streamlit run custom_code_agent.py
   ```
3. Open your browser at `http://localhost:8501` to access the UI.

### Examples
- **Simple Calculation**:
  - Input: `Calculate the sum of 1 to 10`
  - Expected Output: Agent proposes code, user approves, and the result "55" is displayed.

- **Data Visualization**:
  - Input: `Load the Iris dataset and create a line chart of sepal length`
  - Expected Output: Agent generates code, creates a chart, and displays it with a download option.

- **File Processing**:
  - Upload a `data.txt` file with "hello world".
  - Input: `Count the frequency of words in data.txt`
  - Expected Output: Displays word counts (e.g., "hello: 1, world: 1").

## Configuration
- **Sidebar Options**:
  - **LLM Model**: Switch between `llama3-70b-8192` (powerful, slower) and `llama3-8b-8192` (faster).
  - **Temperature**: Adjust from 0.0 (deterministic) to 1.0 (creative) using the slider.
  - **API Key**: Enter or verify your Groq API key.
  - **New Chat**: Start a fresh conversation.
  - **Conversation Selector**: Switch between saved chats.

## Troubleshooting
- **Application Fails to Start**:
  - Check if `GROQ_API_KEY` is set in `.env` and valid.
  - Verify all dependencies are installed (`pip list`).
  - Ensure no port conflicts (default: 8501).

- **No Output from Agent**:
  - Confirm the agent initializes (look for "Failed to initialize Agent" error).
  - Check the Groq API status at [Groq Status Page](https://status.groq.com/).
  - Review the "Agent's Thought Process" expander for logs.

- **Charts Not Displaying**:
  - Ensure `matplotlib` and `seaborn` are installed.
  - Verify the agent outputs valid JSON (e.g., `PLOT_DATA_JSON_START:...`).

- **Timeout Errors**:
  - Code execution is limited to 60 seconds. Simplify complex tasks or increase timeout in `python_executor.py` if needed.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature-name"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
- **Author**: Manaswi Kamble
- **GitHub**: [man-swi](https://github.com/man-swi)
- **Email**: [kamblemanswi8@gmail.com](mailto:kamblemanswi8@gmail.com)
- For support, open an issue on GitHub or contact the author directly.

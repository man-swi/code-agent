import sys
import subprocess
import os
import re
from typing import Type

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

class PythonCodeExecutorToolInput(BaseModel):
    """Input schema for the PythonCodeExecutorTool."""
    code: str = Field(description="The Python code to execute. Must be a complete, runnable script without markdown.")

class PythonCodeExecutorTool(BaseTool):
    """
    A tool for executing Python code in a sandboxed environment.
    This tool takes a string of Python code, cleans it robustly, validates it,
    and then presents it to the user for approval. If approved, it executes
    the code in a separate process.
    """
    name: str = "python_code_executor"
    description: str = (
        "Executes a given snippet of Python code and returns its standard output and standard error. "
        "Use this tool ONLY for running Python code. "
        "The input 'code' MUST be raw Python code only, without any surrounding text, explanations, or markdown "
        "fences (like ```python or ```). "
        "Ensure the Python code is self-contained and prints any results to standard output (e.g., using `print(result)`)."
        "When this tool is called, the code will be presented to the user for confirmation before execution."
    )
    args_schema: Type[BaseModel] = PythonCodeExecutorToolInput

    def _clean_and_validate(self, code_input: str) -> (str, bool):
        """
        Cleans and validates the input code string with a robust strategy.

        This function performs a multi-step cleaning process:
        1.  It first attempts to find and extract code from within markdown fences
            (```python...``` or ```...```), discarding any surrounding text. This is
            the most reliable method.
        2.  If no markdown block is found, it falls back to stripping the fences from
            the start/end of the entire string.
        3.  Finally, it uses `compile()` as a definitive check for valid Python syntax.

        Returns:
            A tuple containing the cleaned code (or an error message) and a boolean
            indicating if the code is valid.
        """
        if not isinstance(code_input, str):
            return "Error: Input code must be a string.", False

        code = code_input

        # **Strategy 1: Find and extract a markdown code block.**
        # This is the most robust way to handle LLM-generated code.
        code_block_match = re.search(r"```(?:python)?\s*\n(.*?)\n```", code, re.DOTALL)
        if code_block_match:
            # If a block is found, we use its content exclusively.
            cleaned_code = code_block_match.group(1).strip()
        else:
            # **Strategy 2: Fallback for code without markdown fences.**
            # This is less robust but handles cases where the LLM forgets the fences.
            # We aggressively strip fences and any conversational lines.
            
            # First, strip potential markdown fences from the whole block
            temp_code = re.sub(r"^\s*```(?:python)?\s*", "", code)
            temp_code = re.sub(r"```\s*$", "", temp_code)
            
            # Split into lines and filter out common non-code phrases
            lines = temp_code.strip().split('\n')
            code_lines = []
            non_code_patterns = [
                re.compile(r"^\s*(here's|here is|the following is|please confirm|certainly|of course|i've prepared|please find below|let me know)\b.*$", re.IGNORECASE),
                re.compile(r"^\s*\.\.\.\s*$"), # Ellipsis line
                re.compile(r"^\s*please confirm the execution of this code\s*$", re.IGNORECASE)
            ]
            
            for line in lines:
                is_non_code = False
                for pattern in non_code_patterns:
                    if pattern.match(line):
                        is_non_code = True
                        break
                if not is_non_code:
                    code_lines.append(line)
            
            cleaned_code = "\n".join(code_lines).strip()


        if not cleaned_code:
            return "Error: No valid Python code found after cleaning the input.", False

        # **Final Validation with compile()**
        try:
            compile(cleaned_code, '<string>', 'exec')
            return cleaned_code, True
        except SyntaxError as e:
            error_message = f"Syntax Error after cleaning: {e}. The final code block is not valid Python."
            return error_message, False

    def _run(self, code: str) -> str:
        """
        This method is called by the agent to propose code. It cleans and
        validates the code before it's shown to the user.
        """
        cleaned_code, is_valid = self._clean_and_validate(code)
        if not is_valid:
            # Return the error message directly to the agent's observation
            return cleaned_code
        return cleaned_code

    def execute_code_after_approval(self, raw_code_input: str) -> str:
        """
        Executes the Python code after user approval from the Streamlit UI.
        """
        cleaned_code, is_valid = self._clean_and_validate(raw_code_input)

        if not is_valid:
            return f"Standard Error:\n{cleaned_code}"

        initial_files = set(os.listdir('.'))
        try:
            process = subprocess.run(
                [sys.executable, '-c', cleaned_code],
                capture_output=True, text=True, timeout=60, check=False
            )

            output_parts = []
            if process.stdout:
                output_parts.append(f"Standard Output:\n{process.stdout.strip()}")
            if process.stderr:
                output_parts.append(f"Standard Error:\n{process.stderr.strip()}")
            result_message = "\n".join(output_parts) or "Code executed with no output."

            final_files = set(os.listdir('.'))
            new_files = final_files - initial_files
            detected_files = [f for f in new_files if os.path.isfile(f) and not f.startswith('.')]
            if detected_files:
                result_message += f"\n\nFiles created during execution: {', '.join(sorted(detected_files))}"

            return result_message.strip()

        except subprocess.TimeoutExpired:
            return "Standard Error:\nExecution timed out after 60 seconds."
        except Exception as e:
            return f"Standard Error:\nAn unexpected error occurred during execution: {str(e)}"

    async def _arun(self, code: str) -> str:
        """Asynchronous version of the run method."""
        return self._run(code)
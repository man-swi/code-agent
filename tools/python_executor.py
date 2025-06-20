import sys
import subprocess
import os
import re
from typing import Type

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

class PythonCodeExecutorToolInput(BaseModel):
    code: str = Field(description="The Python code to execute. It should be a complete, runnable script without any markdown formatting.")

class PythonCodeExecutorTool(BaseTool):
    name: str = "python_code_executor"
    description: str = (
        "Executes a given snippet of Python code and returns its standard output and standard error. "
        "Use this tool ONLY for running Python code. "
        "The input 'code' MUST be raw Python code only, without any surrounding text, explanations, or markdown "
        "fences (like ```python or ```). "
        "Ensure the Python code is self-contained and prints any results to standard output (e.g., using `print(result)`)."
        "When this tool is called, the code will be presented to the user for confirmation before execution."
        "If the code saves a file (e.g., image, GIF), mention the filename in the output so the user knows where to find it."
    )
    args_schema: Type[BaseModel] = PythonCodeExecutorToolInput

    def _clean_and_validate(self, code_input: str) -> (str, bool):
        """
        The definitive cleaning and validation function.
        Returns a tuple: (cleaned_code, is_valid).
        """
        if not isinstance(code_input, str):
            return "Error: Input code must be a string.", False

        # 1. Strip markdown fences
        code = re.sub(r"^\s*```(?:python)?\s*\n", "", code_input)
        code = re.sub(r"\n\s*```\s*$", "", code)
        code = code.strip()

        # 2. Split into lines and filter out conversational text and ellipsis
        lines = code.split('\n')
        code_lines = []
        
        # This regex is now more robust to catch leading conversational phrases
        conversational_pattern = re.compile(
            r"^\s*(here's|here is|the following is|please confirm|certainly|of course|sure, this code|i've prepared|please find below|let me know)\b.*$",
            re.IGNORECASE
        )

        for line in lines:
            stripped_line = line.strip()
            # Explicitly remove ellipsis lines and conversational lines
            if stripped_line == '...':
                continue
            if conversational_pattern.match(stripped_line):
                continue
            code_lines.append(line)

        cleaned_code = "\n".join(code_lines).strip()

        if not cleaned_code:
            return "Error: No valid Python code found after cleaning.", False

        # 3. Final validation with compile()
        try:
            compile(cleaned_code, '<string>', 'exec')
            return cleaned_code, True
        except SyntaxError as e:
            error_message = f"Syntax Error in generated code: {e}. The code is not valid Python. Please regenerate the code."
            return error_message, False

    def _run(self, code: str) -> str:
        """
        This is called by the agent to PROPOSE the code. We just clean it.
        The UI will show this cleaned code to the user.
        """
        cleaned_code, is_valid = self._clean_and_validate(code)
        if not is_valid:
            # If the code is invalid from the start, we return the error
            # so the agent can see its mistake immediately.
            return cleaned_code
        return cleaned_code

    def execute_code_after_approval(self, raw_code_input: str) -> str:
        """This runs AFTER the user clicks 'Approve & Execute'."""
        cleaned_code, is_valid = self._clean_and_validate(raw_code_input)

        if not is_valid:
            # This provides the error message directly to the user's view.
            return f"Standard Error:\n{cleaned_code}"

        # Now, execute the validated code
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

            result_message = "\n".join(output_parts)
            if not result_message:
                result_message = "Code executed successfully with no output."

            new_files = set(os.listdir('.')) - initial_files
            detected_files = [f for f in new_files if os.path.isfile(f) and not f.startswith('.')]
            if detected_files:
                result_message += f"\n\nFiles created during execution: {', '.join(sorted(detected_files))}"

            return result_message.strip()

        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out after 60 seconds."
        except Exception as e:
            return f"An unexpected error occurred during execution: {str(e)}"

    async def _arun(self, code: str) -> str:
        return self._run(code)
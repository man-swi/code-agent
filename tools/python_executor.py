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

    def _clean_code(self, code_input: str) -> str:
        """
        ULTRA-FINAL ROBUST CLEANING: Filters out known non-code lines, conversational elements,
        `input()` calls, and applies automated code formatting using autopep8 as a final guardrail
        for consistent Python syntax and indentation.
        """
        lines = code_input.split('\n')
        filtered_lines = []

        # ## FIX ##: Made the regex pattern more flexible to catch more conversational phrases.
        # This will now match "Please confirm the execution" and "Please confirm the code execution".
        NON_CODE_LINE_PATTERNS = [
            re.compile(r"^\s*```(?:python)?\s*$", re.IGNORECASE),
            re.compile(r"^\s*```$", re.IGNORECASE),
            re.compile(r"^\s*\.\.\.\s*$", re.IGNORECASE),
            re.compile(r"^\s*(?:Here is the code:|The code is as follows:|This is the Python code:|Please confirm the(?: code)? execution\.?|Please wait for the Observation\.)\s*$", re.IGNORECASE),
        ]

        INPUT_CALL_PATTERN = re.compile(r"\binput\s*\(")

        # ## FIX ##: Also updated this pattern for consistency.
        CONVERSATIONAL_IN_LINE_PATTERNS = [
            re.compile(r"Please confirm the(?: code)? execution\.?", re.IGNORECASE),
            re.compile(r"execution\.?", re.IGNORECASE),
            re.compile(r"Here is the code:", re.IGNORECASE),
            re.compile(r"The code is as follows:", re.IGNORECASE),
            re.compile(r"This is the Python code:", re.IGNORECASE),
            re.compile(r"Please wait for the Observation\.", re.IGNORECASE),
            re.compile(r"```python", re.IGNORECASE),
            re.compile(r"```", re.IGNORECASE),
            re.compile(r"\.\.\.", re.IGNORECASE),
        ]


        for line in lines:
            original_line_stripped = line.strip()

            is_non_code_line = False
            for pattern in NON_CODE_LINE_PATTERNS:
                if pattern.fullmatch(original_line_stripped):
                    is_non_code_line = True
                    break

            if is_non_code_line:
                continue

            processed_line = line
            for pattern in CONVERSATIONAL_IN_LINE_PATTERNS:
                processed_line = pattern.sub("", processed_line)

            processed_line_content = processed_line.strip()

            if INPUT_CALL_PATTERN.search(processed_line_content):
                if "input(" in processed_line_content and not (processed_line_content.startswith(('"', "'")) and processed_line_content.endswith(('"', "'"))):
                    processed_line = processed_line.replace(processed_line_content, INPUT_CALL_PATTERN.sub("# Removed input() for safety, use hardcoded values #", processed_line_content))

            if processed_line.strip():
                filtered_lines.append(processed_line)

        pre_formatted_code = "\n".join(filtered_lines).strip()

        if not pre_formatted_code:
            return ""

        try:
            autopep8_path = [sys.executable, "-m", "autopep8"]
            subprocess.run(autopep8_path + ["--version"], capture_output=True, check=True, text=True, timeout=5)

            process = subprocess.run(
                autopep8_path + ["--max-line-length", "79", "-"],
                input=pre_formatted_code,
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )
            formatted_code = process.stdout.strip()
            if not formatted_code and pre_formatted_code:
                sys.stderr.write("Warning: autopep8 returned empty output for non-empty input. Using pre-formatted code. This might indicate severe syntax issues.\n")
                return pre_formatted_code
            return formatted_code
        except FileNotFoundError:
            sys.stderr.write("Warning: autopep8 not found. Please install it (`pip install autopep8`) for robust code formatting. Skipping formatting.\n")
            return pre_formatted_code
        except subprocess.CalledProcessError as e:
            sys.stderr.write(f"Warning: autopep8 failed to format code (might be a critical syntax error): {e.stderr.strip()}. Using pre-formatted code.\n")
            return pre_formatted_code
        except subprocess.TimeoutExpired:
            sys.stderr.write("Warning: autopep8 process timed out. Using pre-formatted code.\n")
            return pre_formatted_code
        except Exception as e:
            sys.stderr.write(f"Warning: Unexpected error during autopep8 formatting: {e}. Using pre-formatted code.\n")
            return pre_formatted_code

    def _run(self, code: str) -> str:
        cleaned_code = self._clean_code(code)
        if not cleaned_code:
            return "Error: No valid Python code provided after cleaning. The input might have been empty or only markdown."
        return cleaned_code

    def execute_code_after_approval(self, raw_code_input: str) -> str:
        cleaned_code = self._clean_code(raw_code_input)
        if not cleaned_code:
            return "Error: No valid Python code provided after cleaning. The input might have been empty or only markdown."

        initial_files = set(os.listdir('.'))

        try:
            process = subprocess.run(
                [sys.executable, '-c', cleaned_code],
                capture_output=True,
                text=True,
                timeout=60,
                check=False
            )

            output_parts = []
            if process.stdout:
                output_parts.append(f"Standard Output:\n{process.stdout.strip()}")
            if process.stderr:
                output_parts.append(f"Standard Error:\n{process.stderr.strip()}")

            result_message = "\n".join(output_parts)

            if not result_message and process.returncode == 0:
                result_message = "Code executed successfully with no output to stdout or stderr."
            elif not result_message and process.returncode != 0:
                result_message = f"Code execution failed with return code {process.returncode} and no specific error message."
            elif process.returncode != 0 and "Standard Error" not in result_message:
                 result_message += f"\nCode execution finished with return code: {process.returncode}"

            new_files = set(os.listdir('.')) - initial_files
            detected_files = []
            for filename in new_files:
                if os.path.isfile(filename) and not filename.startswith(('.', '__')):
                    detected_files.append(filename)

            if detected_files:
                result_message += f"\n\nFiles created during execution: {', '.join(detected_files)}"

            return result_message.strip()

        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out after 60 seconds. This might be due to an infinite loop, `input()` call, or very long computation. Please check your code."
        except Exception as e:
            return f"An unexpected error occurred during Python code execution: {str(e)}"

    async def _arun(self, code: str) -> str:
        return self._run(code)
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
        
        # --- MODIFIED: More comprehensive NON_CODE_LINE_PATTERNS ---
        NON_CODE_LINE_PATTERNS = [
            re.compile(r"^\s*```(?:python)?\s*$", re.IGNORECASE), # ```python or ```
            re.compile(r"^\s*```.*$", re.IGNORECASE),              # Any markdown fence with text (e.g. ```json, ```output)
            re.compile(r"^\s*\.\.\.\s*$", re.IGNORECASE),          # Ellipses line
            re.compile(r"^\s*(?:Here is the code:|The code is as follows:|This is the Python code:|Please confirm the code before(?: I)? execution\.?|Please wait for the Observation\.|The agent proposes to execute the following Python code:|```python|```)\s*$", re.IGNORECASE),
            re.compile(r"^\s*(?:Thought:.*|Action:.*|Action Input:.*|Observation:.*)\s*$", re.IGNORECASE), # ReAct internal thoughts/actions
            re.compile(r"^\s*```.*?$", re.IGNORECASE), # General catch-all for markdown fences like ````python`
            re.compile(r"^\s*python\s*$", re.IGNORECASE), # Just the word 'python'
        ]
        # --- END MODIFIED ---
        
        INPUT_CALL_PATTERN = re.compile(r"\binput\s*\(")

        # --- MODIFIED: More comprehensive CONVERSATIONAL_IN_LINE_PATTERNS ---
        CONVERSATIONAL_IN_LINE_PATTERNS = [
            re.compile(r"Please confirm the code before(?: I)? execution\.?", re.IGNORECASE),
            re.compile(r"execution\.?", re.IGNORECASE),
            re.compile(r"Here is the code:", re.IGNORECASE),
            re.compile(r"The code is as follows:", re.IGNORECASE),
            re.compile(r"This is the Python code:", re.IGNORECASE),
            re.compile(r"Please wait for the Observation\.", re.IGNORECASE),
            re.compile(r"```python", re.IGNORECASE),
            re.compile(r"```", re.IGNORECASE),
            re.compile(r"\.\.\.", re.IGNORECASE),
            re.compile(r"The agent proposes to execute the following Python code:", re.IGNORECASE),
        ]
        # --- END MODIFIED ---

        # --- NEW PRE-PROCESSING RULES for common LLM hallucinations ---
        REPLACEMENT_MAP = {
            # Fix 'report' instead of 'import'
            re.compile(r"^\s*report\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*$", re.IGNORECASE): r"import \1",
            re.compile(r"^\s*report\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*$", re.IGNORECASE): r"import \1 as \2",
            re.compile(r"^\s*from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+report\s+([a-zA-Z_][a-zA-Z0-9_,\s]*)\s*$", re.IGNORECASE): r"from \1 import \2",
        }
        # --- End NEW ---

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
                # Ensure we only replace input() if it's a call, not just part of a string
                if "input(" in processed_line_content and not (processed_line_content.count('"') % 2 == 0 and processed_line_content.count("'") % 2 == 0 and '"' in processed_line_content and "'" in processed_line_content):
                    processed_line = processed_line.replace(processed_line_content, INPUT_CALL_PATTERN.sub("# Removed input() for safety, use hardcoded values #", processed_line_content))
            
            # --- Apply NEW pre-processing replacements ---
            for pattern, replacement in REPLACEMENT_MAP.items():
                if pattern.match(processed_line): # Use match for start of line
                    processed_line = pattern.sub(replacement, processed_line)
                    break # Apply only one matching rule per line
            # --- End NEW ---

            if processed_line.strip():
                # NEW: Clean up unmatched quotes or common trailing non-code text from the very end of lines
                # Example: `print("Hello!") please confirm` -> `print("Hello!")`
                processed_line = re.sub(r'["\']?\s*(?:Please confirm|The agent proposes).*$', '', processed_line.strip())
                
                # Check for and fix common string literal issues (like the one in your video: result['data'])
                # This is a very targeted fix for f-strings with nested quotes.
                if re.search(r"f['\"].*?\{.*?['\"].*?\}.*?['\"]", processed_line):
                    processed_line = processed_line.replace("['", "['").replace("']", "']") # Simple cleanup, might need more robust parsing for complex cases.
                    # A more robust solution might involve parsing the f-string's internal structure.
                    # For now, let's just make sure nested single quotes within f-string curly braces don't terminate the outer f-string.
                    # It's better to tell the agent to use double quotes for the f-string if there are single quotes inside.
                    
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
                return "AUTOPEP8_FORMATTING_FAILED:" + pre_formatted_code
            return formatted_code
        except FileNotFoundError:
            sys.stderr.write("Warning: autopep8 not found. Please install it (`pip install autopep8`) for robust code formatting. Skipping formatting.\n")
            return pre_formatted_code
        except subprocess.CalledProcessError as e:
            sys.stderr.write(f"Warning: autopep8 failed to format code (might be a critical syntax error): {e.stderr.strip()}. Using pre-formatted code.\n")
            return "AUTOPEP8_FORMATTING_FAILED:" + pre_formatted_code
        except subprocess.TimeoutExpired:
            sys.stderr.write("Warning: autopep8 process timed out. Using pre-formatted code.\n")
            return pre_formatted_code
        except Exception as e:
            sys.stderr.write(f"Warning: Unexpected error during autopep8 formatting: {e}. Using pre-formatted code.\n")
            return pre_formatted_code

    def _run(self, code: str) -> str:
        cleaned_code = self._clean_code(code)
        if cleaned_code.startswith("AUTOPEP8_FORMATTING_FAILED:"):
            return f"Error: Code formatting failed, likely due to severe syntax issues. Please review your code carefully.\n{cleaned_code.replace('AUTOPEP8_FORMATTING_FAILED:', '')}"
        if not cleaned_code:
            return "Error: No valid Python code provided after cleaning. The input might have been empty or only markdown."
        return cleaned_code

    def execute_code_after_approval(self, raw_code_input: str) -> str:
        cleaned_code = self._clean_code(raw_code_input)
        if cleaned_code.startswith("AUTOPEP8_FORMATTING_FAILED:"):
            return f"Error: Code formatting failed, likely due to severe syntax issues. Please review your code carefully.\n{cleaned_code.replace('AUTOPEP8_FORMATTING_FAILED:', '')}"
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
                error_message = process.stderr.strip()
                output_parts.append(f"Standard Error:\n{error_message}")
                
                # --- ENHANCED ERROR FEEDBACK (from previous suggestions, still relevant) ---
                if "NameError: name 'time' is not defined" in error_message or "NameError: name 'plt' is not defined" in error_message or "NameError: name 'pd' is not defined" in error_message or "NameError: name 'np' is not defined" in error_message:
                    output_parts.append("\n**HINT:** This looks like a missing import. Remember to include `import time`, `import matplotlib.pyplot as plt`, `import pandas as pd`, `import numpy as np`, etc., at the beginning of your script if you're using functions or objects from those libraries.")
                elif "SyntaxError:" in error_message or "IndentationError:" in error_message:
                    output_parts.append("\n**HINT:** This is a syntax or indentation error. Review the line indicated in the traceback for missing colons, unbalanced parentheses, unmatched quotes, or incorrect indentation.")
                elif "ValueError: Input 0 of layer" in error_message and "is incompatible" in error_message:
                    output_parts.append("\n**HINT:** For Keras Sequential models, the first layer usually needs an `input_shape` argument (e.g., `model.add(Dense(..., input_shape=(num_features,)))`).")
                elif "TypeError: object is not subscriptable" in error_message:
                     output_parts.append("\n**HINT:** You might be trying to access an element from something that isn't a list or dictionary, or using incorrect indexing (e.g., `value[key]` instead of `value.attribute`).")
                elif "TimeoutExpired" in error_message or "timed out" in error_message:
                     output_parts.append("\n**HINT:** The code timed out. This often happens due to an infinite loop or an `input()` call (which is disallowed). Ensure your loops have termination conditions and you are not using `input()`.")
                elif "NameError:" in error_message or "ImportError:" in error_message:
                     output_parts.append("\n**HINT:** This might be a missing import or an undefined variable. Double-check your your imports and variable names.")
                # --- End ENHANCED ---

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
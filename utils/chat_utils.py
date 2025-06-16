import re 

def generate_chat_name(initial_prompt: str) -> str:
    """Generates a concise name for a chat based on its first prompt."""
    if not initial_prompt:
        return "Untitled Chat"
    
    # Clean up and truncate the prompt for a good display name
    name = initial_prompt.strip()
    name = name.replace('\n', ' ') # Replace newlines with spaces for single-line name
    name = re.sub(r'\s+', ' ', name) # Collapse multiple spaces
    
    if len(name) > 30:
        return name[:27] + "..." # Truncate and add ellipsis
    return name
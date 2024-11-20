"""
This is a prompt for the task of converting text into bullet points as a Python list.
"""
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# System Prompt
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """
[Classify Task into Predefined Label]

<prompt_objective>  
Classify the provided task into one of the predefined labels: "work," "home," "errands," or "none."  
</prompt_objective>  

<prompt_rules>  
- ONLY use the labels: "work," "home," "errands," and "none."  
- NEVER create new labels under any circumstances.  
- Assign "none" if the task does not clearly match any of the other three labels.  
- The output must consist of ONLY the label, with no additional text or formatting.  
</prompt_rules>  

<prompt_examples>  
USER: Prepare the quarterly budget report.  
AI: work  

USER: Fix the leaky faucet in the kitchen.  
AI: home  

USER: Pick up groceries from the store.  
AI: errands  

USER: Walk the dog.  
AI: home  

USER: Write a blog post for the company website.  
AI: work  

USER: Watch a movie.  
AI: none  

USER: Attend the parent-teacher conference.  
AI: home  

USER: Schedule an interview with a candidate.  
AI: work  

USER: Take the car for an oil change.  
AI: errands  

USER: Relax and do nothing.  
AI: none  
</prompt_examples> 
"""

# -----------------------------------------------------------------------------
# Pydantic Model
# -----------------------------------------------------------------------------


class OutputModel(BaseModel):
    """
    Pydantic model for the answers.
    """
    label: str

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


def get_messages() -> list[dict]:
    """
    Get the messages.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "{{text}}"},
    ]


def compile_messages(text: str) -> list[dict]:
    """
    Compile the messages with the given context by replacing template variables.

    Args:
        facts: The factual context to inject into the system prompt
        text: The text content for the user message

    Returns:
        list[dict]: List of message dictionaries with populated template variables
    """
    messages = get_messages()

    # Replace template variables in user message
    messages[1]["content"] = messages[1]["content"].replace("{{text}}", text)

    return messages

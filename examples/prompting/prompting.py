# pylint: disable=missing-module-docstring, missing-function-docstring, invalid-name
from dotenv import load_dotenv
from openai import OpenAI

from prompts.task_label import TaskLabelPrompt

load_dotenv()

# -----------------------------------------------------------------------------
# Define clients
# -----------------------------------------------------------------------------

# OpenAI client
openai_client = OpenAI()

# Ollama client
ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# -----------------------------------------------------------------------------
# Define prompts
# -----------------------------------------------------------------------------

task_label_prompt = TaskLabelPrompt()

# -----------------------------------------------------------------------------
# Define functions
# -----------------------------------------------------------------------------

def get_response_from_openai(user_message: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=task_label_prompt.compile_messages(user_message)
    )
    return response.choices[0].message.content

def get_response_from_ollama(user_message: str) -> str:
    response = ollama_client.chat.completions.create(
        model="llama3.1:8b",
        messages=task_label_prompt.compile_messages(user_message)
    )
    return response.choices[0].message.content

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    tasks = [
        "Prepare the quarterly budget report.",
        "Prepare the quarterly sales report.",
        "Prepare the quarterly marketing report.",
        "Read book about AI.",
        "Wash the dishes.",
        "Call the bank to check the credit card balance.",
        "Buy milk.",
    ]
    for task in tasks:
        label = get_response_from_openai(task)
        print(f"{task} -> {label}")

if __name__ == "__main__":
    main()

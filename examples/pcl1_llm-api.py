# pylint: disable=missing-module-docstring, missing-function-docstring, invalid-name
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# OpenAI client

openai_client = OpenAI()

def get_response_from_openai(user_message: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )
    return response.choices[0].message.content

# Ollama client

ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def get_response_from_ollama(user_message: str) -> str:
    response = ollama_client.chat.completions.create(
        model="llama3.1:8b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    message = "What is the capital of Japan?"
    print(get_response_from_ollama(message))

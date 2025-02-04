"""
This script shows how to use Word Embeddings to find relevant sentences.
"""
import os
import json
import csv
from typing import List
from uuid import uuid4
from dotenv import load_dotenv
from tqdm import tqdm

from src.service.embedding_service import EmbeddingService

script_path = os.path.dirname(os.path.abspath(__file__))
parameters_txt_path = os.path.join(script_path, "parameters.txt")
parameters_json_path = os.path.join(script_path, "parameters.json")
tasks_json_path = os.path.join(script_path, "extraction_task_contr.json")
principles_txt_path = os.path.join(script_path, "principles_names.txt")
# Load the environment variables
load_dotenv()

# Define embedding service
embedding_service = EmbeddingService(model="mxbai-embed-large:latest")

def load_matrix_values(file_path: str) -> dict:
    """
    Load the matrix values from a CSV file.
    """
    matrix = {}
    with open(file_path, "r", encoding="utf-8-sig") as file:
        reader = csv.reader(file, delimiter=';')
        for row_index, row in enumerate(reader):
            for col_index, cell in enumerate(row):
                if cell.strip():  # Check if the cell is not empty
                    # Split by comma, strip any extra spaces, and handle multiple spaces
                    matrix[(row_index + 1, col_index + 1)] = [int(x.strip()) for x in cell.replace(' ', '').split(',') if x.strip()]
    return matrix

def find_principles(positive_index: int, negative_index: int, matrix: dict) -> List[int]:
    """
    Find the inventive principles for the given parameter indices using the matrix.
    """
    return matrix.get((positive_index, negative_index), [])

def load_principles(file_path: str) -> List[str]:
    """
    Load the principles from a text file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        principles = [line.strip() for line in file.readlines()]
    return principles

def embed_parameters(parameters: List[str]) -> dict:
    """
    Embed the TRIZ standard parameters.
    """
    output_list = []
    progress_bar = tqdm(enumerate(parameters), total=len(parameters), desc="Embedding parameters")
    for ind, parameter in progress_bar:
        output_list.append(
            {
                "uuid": str(uuid4()),
                "index": ind,
                "parameter": parameter,
                "embedding": embedding_service.create_embedding(parameter),
            }
        )
    return output_list

def search_parameters(query: str, parameters: List[dict], n: int = 1) -> List[dict]:
    """
    Search for the closest parameters.
    """
    query_embedding = embedding_service.create_embedding(query)
    distances = embedding_service.find_n_closest(
        query_vector=query_embedding,
        embeddings=[param["embedding"] for param in parameters],
        n=n,
    )
    return [parameters[dist["index"]] for dist in distances]

def remove_embedding_fields(parameters):
    """
    Remove the 'embedding' field from each parameter in the list.
    """
    for param in parameters:
        param.pop("embedding", None)


def main():
    """
    Main function.
    """
    # Load tasks from JSON file
    if os.path.exists(tasks_json_path):
        print(f"Loading tasks from {tasks_json_path}...")
        try:
            with open(tasks_json_path, "r", encoding="utf-8") as file:
                tasks = json.load(file)
            print(f"Successfully loaded {len(tasks)} tasks")
        except Exception as e: # pylint: disable=broad-except
            print(f"Failed to load tasks: {e}")
            return
    else:
        print(f"Tasks file {tasks_json_path} not found.")
        return

    # Check if embeddings file exists
    if os.path.exists(parameters_json_path):
        print("Loading existing embeddings from JSON file...")
        try:
            with open(parameters_json_path, "r", encoding='utf-8') as file:
                parameters = json.load(file)
            print(f"Successfully loaded {len(parameters)} embeddings")
        except Exception as e: # pylint: disable=broad-except
            print(f"Failed to load existing embeddings: {e}")
            return
    else:
        print("Embeddings file not found. Generating new embeddings...")
        # Load the TRIZ standard parameters
        try:
            with open(parameters_txt_path, "r", encoding='utf-8') as file:
                parameters_txt = [line.strip() for line in file.readlines()]
            print(f"Successfully loaded {len(parameters_txt)} parameters from text file")
        except Exception as e: # pylint: disable=broad-except
            print(f"Failed to load parameters from text file: {e}")
            return

        # Generate and save embeddings
        try:
            parameters = embed_parameters(parameters_txt)
            with open(parameters_json_path, "w", encoding='utf-8') as file:
                json.dump(parameters, file, indent=4)
            print(f"Successfully generated and saved {len(parameters)} embeddings to {parameters_json_path}")
        except Exception as e: # pylint: disable=broad-except
            print(f"Failed to generate and save embeddings: {e}")
            return

    # Load matrix values
    matrix_values_path = os.path.join(script_path, "matrix_values.csv")
    matrix = load_matrix_values(matrix_values_path)

    # Load principles
    principles = load_principles(principles_txt_path)

    # Embed positive and negative fields and search for closest parameters
    for task in tqdm(tasks, desc="Embedding tasks"):
        task["closest_positive_parameters"] = search_parameters(task["postive"], parameters)
        task["closest_negative_parameters"] = search_parameters(task["negative"], parameters)

        positive_index = task["closest_positive_parameters"][0]["index"] + 1
        negative_index = task["closest_negative_parameters"][0]["index"] + 1

        principle_indices = find_principles(positive_index, negative_index, matrix)
        task["Principles"] = [principles[i - 1] for i in principle_indices]  # Convert indices to principles

    # Remove embedding fields from the parameters
    for task in tasks:
        remove_embedding_fields(task["closest_positive_parameters"])
        remove_embedding_fields(task["closest_negative_parameters"])

    # Save the results to a JSON file
    output_path = os.path.join(script_path, "tasks_with_embeddings_and_principles.json")
    print(f"Saving results to {output_path}")
    try:
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(tasks, file, indent=4)
        print(f"Successfully saved tasks with principles to {output_path}")
    except Exception as e: # pylint: disable=broad-except
        print(f"Failed to save tasks with principles: {e}")

    # Print the required information to the console
    for task in tasks:
        print(f"Description: {task['description']}\n")
        print(f"Positive Parameters: {[param['parameter'] for param in task['closest_positive_parameters']]}\n")
        print(f"Negative Parameters: {[param['parameter'] for param in task['closest_negative_parameters']]}\n")
        print(f"Applicable Principles: {task['Principles']}\n")
        print("-" * 80)

if __name__ == '__main__':
    main()

"""
This script shows how to use Word Embeddings to find relevant sentences.
"""
import os
import json
import logging
from typing import List
from uuid import uuid4
from dotenv import load_dotenv
from tqdm import tqdm

from service.embedding_service import EmbeddingService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embeddings.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

script_path = os.path.dirname(os.path.abspath(__file__))
parameters_txt_path = os.path.join(script_path, "parameters.txt")

# Load the environment variables
load_dotenv()

# Define embedding service
embedding_service = EmbeddingService(model="mxbai-embed-large:335m")
logger.info("Initialized embedding service with model: mxbai-embed-large:335m")

def embed_parameters(parameters: List[str]) -> dict:
    """
    Embed the TRIZ standard parameters.
    """
    logger.info("Starting to embed %s parameters", len(parameters))
    output_list = []
    for ind, parameter in tqdm(enumerate(parameters), desc="Embedding parameters"):
        output_list.append(
            {
                "uuid": str(uuid4()),
                "index": ind,
                "parameter": parameter,
                "embedding": embedding_service.create_embedding(parameter),
            }
        )
    logger.info("Finished embedding parameters")
    return output_list

def search_parameters(query: str, parameters: List[dict], n: int = 3) -> List[dict]:
    """
    Search for the closest parameters.
    """
    logger.info("Searching for %s closest parameters to query: %s", n, query)
    query_embedding = embedding_service.create_embedding(query)
    distances = embedding_service.find_n_closest(
        query_vector=query_embedding,
        embeddings=[param["embedding"] for param in parameters],
        n=n,
    )
    return [parameters[dist["index"]] for dist in distances]

def main():
    """
    Main function.
    """
    logger.info("Starting main function")
    query = "Honeycomb panel used as a core material in the trailer's platform reduces vibrations."

    # Load the TRIZ standard parameters
    try:
        with open(parameters_txt_path, "r", encoding='utf-8') as file:
            parameters_txt = [line.strip() for line in file.readlines()]
        logger.info("Loaded %s parameters from file", len(parameters_txt))
    except Exception as e: # pylint: disable=broad-except
        logger.error("Failed to load parameters file: %s", e)
        return

    # Prepare .json file with embeddings
    try:
        parameters = embed_parameters(parameters_txt)
        with open("parameters.json", "w", encoding='utf-8') as file:
            json.dump(parameters, file, indent=4)
        logger.info("Successfully saved embeddings to parameters.json")
    except Exception as e: # pylint: disable=broad-except
        logger.error("Failed to save embeddings: %s", e)
        return

    # Search for the closest parameters
    closest_parameters = search_parameters(query, parameters)
    print("Closest parameters to the query:")
    for param in closest_parameters:
        print(param["parameter"])

if __name__ == '__main__':
    main()

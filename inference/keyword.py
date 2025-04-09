from prompts.utils import format_keyword_prompt
from inference.utils import process_with_llm, process_files, create_llm_client
from typing import List
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

def extract_keywords(message: str, game: str, model = "meta-llama/Llama-3.1-8B-Instruct", api_key = os.getenv("HF_TOKEN")) -> List[str]:
    """
    Extract keywords from a message using the keyword extraction prompt.

    Args:
        message: The message to extract keywords from
        game: The game the message is from

    Returns:
        List of keywords extracted from the message
    """
    # Create the client correctly
    client = create_llm_client(model, api_key)
    
    # Format the prompt
    prompt_data = format_keyword_prompt(message, game)
    
    # Extract the prompt string from the result dictionary
    prompt = prompt_data["prompt"] if isinstance(prompt_data, dict) and "prompt" in prompt_data else prompt_data
    
    # Process with LLM
    response = process_with_llm(client, prompt)
    
    return response

def extract_keywords_from_data(model, api_key, data_path: str, output_path: str):
    """
    Extract keywords from all messages in the data directory and save the results to a CSV file.

    Args:
        data_path: The path to the data directory
        output_path: The path to save the results to
    """
    client = create_llm_client(model, api_key)
    # Define a simple formatting function that maps CSV columns to prompt parameters
    def format_row_for_keyword(row, game=None):
        # Use either the game from the row or the one provided as argument
        game_value = game if game else row.get('game', '')
        return format_keyword_prompt(message=row['message'], game=game_value)

    # Use it in your processing
    process_files(
        client=client,
        input_folder=data_path,
        output_folder=output_path,
        format_prompt_function=format_row_for_keyword
    )



 


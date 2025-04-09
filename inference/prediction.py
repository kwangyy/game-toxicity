from prompts.utils import format_prediction_prompt
from inference.utils import process_with_llm, process_files, create_llm_client
from typing import Dict, Any
from dotenv import load_dotenv
import os
from openai import OpenAI
import json

load_dotenv()

def predict_toxicity(
    message: str,
    model = "meta-llama/Llama-3.1-8B-Instruct", 
    api_key = os.getenv("HF_TOKEN")
) -> Dict[str, Any]:
    """
    Predict the toxicity of a message.

    Args:
        message: The message to analyze
        model: The model to use for prediction
        api_key: The Hugging Face API key

    Returns:
        Dictionary containing the toxicity prediction
    """
    # Create the client correctly
    client = create_llm_client(model, api_key)
    
    # Format the prompt
    prompt_data = format_prediction_prompt(original_message=message)
    
    # Extract the prompt string from the result dictionary
    prompt = prompt_data["prompt"] if isinstance(prompt_data, dict) and "prompt" in prompt_data else prompt_data
    
    # Process with LLM
    response = process_with_llm(client, prompt)
    
    return response

def predict_toxicity_from_data(
    model,
    api_key,
    data_path: str, 
    output_path: str
):
    """
    Predict toxicity for messages in a CSV file and save the results.

    Args:
        model: The model to use for prediction
        api_key: The Hugging Face API key
        data_path: Path to the folder containing the CSV file
        output_path: Path to save the processed CSV file
    """
    client = create_llm_client(model, api_key)
    
    # Define formatting function for each row
    def format_row_for_prediction(row):
        return format_prediction_prompt(original_message=row['message'])
    
    # Process the CSV file
    process_files(
        client=client,
        input_folder=data_path,
        output_folder=output_path,
        format_prompt_function=format_row_for_prediction
    ) 

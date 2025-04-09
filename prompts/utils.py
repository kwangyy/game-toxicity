import os
from typing import Dict, Any, List, Optional
import random

def load_prompt(prompt_name: str) -> str:
    """
    Load a prompt template from a file in the prompts directory.
    
    Args:
        prompt_name: Name of the prompt file (without .txt extension)
        
    Returns:
        The prompt template as string
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, f"{prompt_name}.txt")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

def format_keyword_prompt(message: str, game: str) -> Dict[str, str]:
    """
    Format the keyword extraction prompt with the message and game.
    
    Args:
        message: The message to extract keywords from
        game: The game the message is from
        
    Returns:
        Dictionary with formatted prompt ready for the LLM
    """
    template = load_prompt("keyword")
    # Add context about the game and message
    formatted_prompt = f"{template}\n\nGame: {game}\nMessage: \"{message}\"\n"
    return {"prompt": formatted_prompt}

def format_prediction_prompt(original_message: str, definitions: dict = None) -> str:
    """
    Format the prompt for toxicity prediction with gaming context definitions.
    
    Args:
        original_message: The message to analyze
        definitions: Dictionary of gaming terms and their definitions across namespaces
        
    Returns:
        Formatted prompt string
    """
    # Read the base prompt template
    with open('prompts/prediction.txt', 'r') as f:
        base_prompt = f.read()
    
    # Add gaming context section before the examples
    context_section = ""
    if definitions:
        context_section = "\nGaming Context (consider these definitions when analyzing):\n"
        for keyword, namespace_defs in definitions.items():
            context_section += f"\n'{keyword}' meanings:\n"
            for namespace, info in namespace_defs.items():
                context_section += f"- {namespace}: {info['definition']}\n"
        context_section += "\n"
    
    # Insert the context section before the examples and add the message to analyze
    prompt = base_prompt + context_section + f"\nAnalyze this message:\n{{\n  \"message\": \"{original_message}\"\n}}"
    
    return prompt


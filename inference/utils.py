import re 
import os 
import json
from typing import Optional, Dict, Any
from huggingface_hub import InferenceClient
import csv


def extract_json_from_response(response_text):
    """
    Extracts JSON from response text, handling code blocks and multiline content.
    Removes comments before parsing.
    """
    def remove_comments(json_str):
        json_str = re.sub(r'#.*$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        return json_str

    # First try to find JSON within code blocks
    code_block_pattern = r'```json\s*([\s\S]*?)\s*```'
    code_block_match = re.search(code_block_pattern, response_text)
    
    if code_block_match:
        try:
            json_str = code_block_match.group(1)
            json_str = remove_comments(json_str)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # If no code block or invalid JSON, try general JSON pattern
    json_pattern = r'(?s)\{.*?\}(?=\s*$)'
    json_match = re.search(json_pattern, response_text)
    
    if json_match:
        try:
            json_str = json_match.group(0)
            json_str = remove_comments(json_str)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    return None

def create_llm_client(model_name, api_key):
    """Creates a new LLM client with the specified model and API key."""
    return InferenceClient(
        model=model_name,
        token=api_key
    )

def process_with_llm(
    client,
    prompt,
    temperature = 0.6,
    max_tokens = 4096,
    top_p = 0.7,
    json_output = True
):
    """
    Process a pre-populated prompt with the LLM client.
    
    Args:
        client: The LLM client to use
        prompt: The pre-populated prompt string to send to the LLM
        temperature: Controls randomness in generation
        max_tokens: Maximum number of tokens to generate
        top_p: Nucleus sampling parameter
        json_output: Whether to attempt to extract JSON from the response
        
    Returns:
        Dictionary containing the LLM response
    """
    try:
        print(f"Sending prompt to LLM: {prompt}")
        
        # Create the completion
        completion = client.chat.completions.create(
            model=client.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        
        # Get the response text from the completion
        response_text = completion.choices[0].message.content
        print(f"Raw LLM response: {response_text}")
        
        if json_output:
            try:
                # Try to parse as JSON first
                return json.loads(response_text)
            except json.JSONDecodeError:
                # If not valid JSON, extract JSON from the text
                extracted_json = extract_json_from_response(response_text)
                if extracted_json is not None:
                    return extracted_json
                # If no JSON found, return as content
                return {"content": response_text}
        
        return {"content": response_text}
        
    except Exception as e:
        print(f"Error in process_with_llm: {str(e)}")
        if json_output:
            return {"error": str(e)}
        return {"content": f"Error: {str(e)}"}

def process_files(
    client,
    input_folder,
    output_folder,
    format_prompt_function,
    **format_kwargs
):
    """
    Process a CSV file row by row, applying a prompt formatting function to each row.
    
    Args:
        client: The LLM client to use
        input_folder: Folder containing CSV files
        output_folder: Folder to save processed CSV files
        format_prompt_function: Function that takes a row and returns a formatted prompt
        **format_kwargs: Additional keyword arguments to pass to the formatting function
    """
    if not os.path.exists(input_folder):
        print(f"The folder '{input_folder}' does not exist.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Find the first CSV file in the folder
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in '{input_folder}'.")
        return
        
    csv_file = csv_files[0]
    csv_file_path = os.path.join(input_folder, csv_file)
    base_name = os.path.splitext(csv_file)[0]
    output_csv_path = os.path.join(output_folder, f"{base_name}_processed.csv")
    
    results = []
    
    # Read all rows from the CSV file
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        fieldnames = csv_reader.fieldnames + ['llm_result']
        rows = list(csv_reader)
    
    # Process each row with the LLM
    for i, row in enumerate(rows):
        print(f"Processing row {i+1} from '{csv_file}'")
        
        # Format the prompt using the provided function
        prompt_result = format_prompt_function(row, **format_kwargs)
        
        # Extract prompt from result (assumes function returns dict with "prompt" key)
        if isinstance(prompt_result, dict) and "prompt" in prompt_result:
            populated_prompt = prompt_result["prompt"]
        else:
            populated_prompt = prompt_result
        
        # Process the prompt with the LLM
        result = process_with_llm(client, populated_prompt)
        
        # Add the result to the row
        if isinstance(result, dict) and 'content' in result:
            row['llm_result'] = result['content']
        else:
            row['llm_result'] = json.dumps(result) if result else str(result)
        
        results.append({"row": i+1, "original_data": dict(row), "prediction": result})
    
    # Write the updated rows to a new CSV file
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as file:
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    
    print(f"Processed {len(rows)} rows from '{csv_file}' and saved results to '{os.path.basename(output_csv_path)}'.")
    
    return results
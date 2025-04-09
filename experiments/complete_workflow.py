import os
import sys
import csv
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import json
from pinecone import Pinecone
from openai import OpenAI

# Load environment variables
load_dotenv()

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import necessary modules
from inference.keyword import extract_keywords
from inference.prediction import predict_toxicity
from prompts.utils import format_keyword_prompt, format_prediction_prompt
from vector.query import match_keywords_to_definitions  # Import the new function

def process_dataset_with_complete_workflow(
    input_file,
    output_file,
    model="meta-llama/Llama-3.1-8B-Instruct",
    api_key=os.getenv("HF_TOKEN"),
    limit=None
):
    """
    Process the dataset using the complete workflow (keywords + definitions + prediction).
    Uses RAG-based dictionary for definitions.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the output CSV file
        model: The model to use for all steps
        api_key: The API key for the model
        limit: Optional limit on the number of rows to process (for testing)
    """
    print(f"Starting complete workflow processing on: {input_file}")
    print(f"Using model: {model}")
    
    # Read only the necessary columns from the dataset
    df = pd.read_csv(input_file)
    
    # Limit rows if specified (for testing)
    if limit:
        df = df.head(limit)
        print(f"Limited processing to {limit} rows for testing")
    
    # Initialize new columns
    df['extracted_keywords'] = None
    df['used_definitions'] = None
    df['model_prediction'] = None
    
    # Keep track of progress
    total_rows = len(df)
    print(f"Processing {total_rows} messages...")
    
    # Process each message
    for idx, row in df.iterrows():
        message = row['message']
        game = row.get('game', 'Unknown Game')
        region = row.get('location')  # Get region if available
        
        # Update progress
        progress = (idx + 1) / total_rows * 100
        print(f"Processing message {idx+1}/{total_rows} ({progress:.1f}%)...")
        
        try:
            # Step 1: Extract keywords
            print(f"  Extracting keywords...")
            # Format the keyword prompt
            keyword_prompt = format_keyword_prompt(
                message=message,
                game=game
            )
            
            # Get keywords using Hugging Face
            keyword_result = extract_keywords(
                message=keyword_prompt,
                game=game,
                model=model,
                api_key=api_key
            )
            
            # Parse and store the keywords
            if isinstance(keyword_result, dict):
                if 'keywords' in keyword_result:  # Direct access to keywords array
                    keywords = keyword_result['keywords']
                    df.at[idx, 'extracted_keywords'] = str(keywords)
                elif 'content' in keyword_result:  # Fallback to content parsing
                    content = keyword_result['content']
                    df.at[idx, 'extracted_keywords'] = content
                    if isinstance(content, str) and content.startswith('[') and content.endswith(']'):
                        keywords = json.loads(content.replace("'", "\""))
                    else:
                        keywords = []
            else:
                df.at[idx, 'extracted_keywords'] = str(keyword_result)
                keywords = []

            
            # Step 2: Get relevant definitions using RAG
            print(f"  Retrieving definitions...")
            if keywords:
                # Use the match_keywords_to_definitions function
                definition_matches = match_keywords_to_definitions(
                    keywords=keywords,
                    game=game,
                    region=region,
                    threshold=0.7  # Adjust threshold as needed
                )
                
                # Format definitions for prompt
                print(definition_matches)
                definitions = {}
                for keyword, matches in definition_matches.items():
                    # Keep all matches across namespaces
                    definitions[keyword] = {}
                    for namespace, match_info in matches.items():
                        definitions[keyword][namespace] = {
                            'matched_word': match_info['matched_word'],
                            'definition': match_info['definition'],
                            'score': match_info['score']
                        }
            else:
                definitions = {}
            
            # Store the definitions used
            df.at[idx, 'used_definitions'] = str(definitions)
            
            # Step 3: Predict toxicity using the definitions
            print(f"  Predicting toxicity...")
            # Format the prediction prompt
            prediction_prompt = format_prediction_prompt(
                original_message=message,
                definitions=definitions  # Add the definitions we gathered
            )
            
            # Get prediction
            prediction_result = predict_toxicity(
                message=prediction_prompt,
                model=model,
                api_key=api_key
            )
            
            # Extract prediction
            if isinstance(prediction_result, dict) and 'content' in prediction_result:
                prediction = prediction_result['content']
            else:
                prediction = str(prediction_result)
                
            df.at[idx, 'model_prediction'] = prediction
            
        except Exception as e:
            print(f"\nError processing message {idx+1}: {str(e)}")
            if df.at[idx, 'extracted_keywords'] is None:
                df.at[idx, 'extracted_keywords'] = f"ERROR: {str(e)}"
            if df.at[idx, 'used_definitions'] is None:
                df.at[idx, 'used_definitions'] = f"ERROR: {str(e)}"
            if df.at[idx, 'model_prediction'] is None:
                df.at[idx, 'model_prediction'] = f"ERROR: {str(e)}"
        
        # Save intermediate results
        if (idx + 1) % 10 == 0 or idx == len(df) - 1:
            df.to_csv(output_file, index=False)
            print(f"  Saved intermediate results to: {output_file}")
    
    print("\nWorkflow complete!")
    df.to_csv(output_file, index=False)
    
    # Calculate stats
    try:
        predicted_toxic = df['model_prediction'].str.contains('toxic|Toxic', na=False).sum()
        print(f"\nQuick stats:")
        print(f"Total messages: {len(df)}")
        print(f"Predicted toxic: {predicted_toxic} ({predicted_toxic/len(df)*100:.1f}%)")
    except Exception as e:
        print(f"Could not calculate stats: {str(e)}")
    
    return df

if __name__ == "__main__":
    input_file = os.path.join(project_root, "process", "processed_data.csv")
    output_dir = os.path.join(project_root, "experiments", "results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "complete_workflow_results.csv")
    
    process_dataset_with_complete_workflow(
        input_file=input_file,
        output_file=output_file,
        # limit=5  # Uncomment for testing
    )
import os
import sys
import csv
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import necessary modules
from inference.prediction import predict_toxicity
from prompts.utils import format_prediction_prompt

def process_dataset_with_base_prediction(
    input_file,
    output_file,
    model="meta-llama/Llama-3.1-8B-Instruct",
    api_key=os.getenv("HF_TOKEN"),
    limit=None
):
    """
    Process the dataset using only the base prediction model.
    Preserves all original columns and adds model prediction.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the output CSV file
        model: The model to use for prediction
        api_key: The API key for the model
        limit: Optional limit on the number of rows to process (for testing)
    """
    print(f"Starting base prediction processing on: {input_file}")
    print(f"Using model: {model}")
    
    # Read all columns from the dataset
    df = pd.read_csv(input_file)
    
    # Limit rows if specified (for testing)
    if limit:
        df = df.head(limit)
        print(f"Limited processing to {limit} rows for testing")
    
    # Initialize new column for model predictions
    df['model_prediction'] = None
    
    # Keep track of progress
    total_rows = len(df)
    print(f"Processing {total_rows} messages...")
    
    # Process each message
    for idx, row in df.iterrows():
        message = row['message']
        
        # Update progress
        progress = (idx + 1) / total_rows * 100
        print(f"Processing message {idx+1}/{total_rows} ({progress:.1f}%)...")
        
        try:
            # Format the prompt
            prompt_data = format_prediction_prompt(original_message=message)
            
            # Get prediction
            result = predict_toxicity(
                message=prompt_data,
                model=model,
                api_key=api_key
            )
            
            # Extract prediction content
            if isinstance(result, dict) and 'content' in result:
                prediction = result['content']
            else:
                prediction = str(result)
            
            # Store prediction
            df.at[idx, 'model_prediction'] = prediction
            
        except Exception as e:
            print(f"\nError processing message {idx+1}: {str(e)}")
            df.at[idx, 'model_prediction'] = f"ERROR: {str(e)}"
        
        # Save intermediate results every 10 rows
        if (idx + 1) % 10 == 0 or idx == len(df) - 1:
            df.to_csv(output_file, index=False)
            print(f"  Saved intermediate results to: {output_file}")
    
    print("\nPrediction complete!")
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
    output_file = os.path.join(output_dir, "base_prediction_results.csv")
    
    process_dataset_with_base_prediction(
        input_file=input_file,
        output_file=output_file,
        # limit=5  # Uncomment for testing
    )
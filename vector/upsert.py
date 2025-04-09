import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
from tqdm.auto import tqdm
from Levenshtein import distance

# Load environment variables
load_dotenv()

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

def initialize_pinecone() -> Pinecone:
    """Initialize Pinecone connection and create index if it doesn't exist."""
    api_key = os.getenv("PINECONE_API_KEY")
    
    if not api_key:
        raise ValueError("PINECONE_API_KEY must be set in .env")
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key)
    
    # Create index if it doesn't exist
    index_name = "dsa4288"
    existing_indexes = [index["name"] for index in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embedding dimension
            metric="cosine",
            spec={"cloud": "aws", "region": "us-east-1"}
        )
    
    return pc.Index(index_name)

def load_definitions(file_path: str) -> List[Dict[str, str]]:
    """Load words and definitions from CSV file."""
    df = pd.read_csv(file_path)
    return [
        {'word': row['word'], 'definition': row['meaning']}
        for _, row in df.iterrows()
    ]

def get_embeddings(words: List[str]) -> np.ndarray:
    """Get OpenAI embeddings for a list of words."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=words
    )
    return np.array([item.embedding for item in response.data])

def upsert_definitions(
    index: Pinecone,
    word_data: List[Dict[str, str]],
    embeddings: np.ndarray,
    namespace: str = None,
    batch_size: int = 32
) -> None:
    """Upsert words, definitions, and embeddings to Pinecone in batches."""
    for i in tqdm(range(0, len(word_data), batch_size)):
        # set end position of batch
        i_end = min(i + batch_size, len(word_data))
        
        # get batch of data
        batch_data = word_data[i:i_end]
        batch_embeddings = embeddings[i:i_end]
        
        # create IDs for batch
        ids_batch = [f"{namespace or 'general'}_{n}" for n in range(i, i_end)]
        
        # create metadata
        meta = [{'word': data['word'], 'definition': data['definition']} 
               for data in batch_data]
        
        # create upsert data using zip
        to_upsert = list(zip(ids_batch, batch_embeddings.tolist(), meta))
        
        # upsert to Pinecone
        if namespace:
            index.upsert(vectors=to_upsert, namespace=namespace)
        else:
            index.upsert(vectors=to_upsert)

def process_definitions_folder():
    """Process all definition files in the definitions folder."""
    # Initialize Pinecone
    print("Initializing Pinecone...")
    index = initialize_pinecone()
    
    # Get the definitions directory
    definitions_dir = os.path.join(current_dir, "definitions")
    
    # Process each CSV file
    for csv_file in os.listdir(definitions_dir):
        if not csv_file.endswith(".csv"):
            continue
            
        # Get namespace from filename
        namespace = os.path.splitext(csv_file)[0]
        file_path = os.path.join(definitions_dir, csv_file)
        
        print(f"\nProcessing {csv_file}...")
        
        # Load definitions
        word_data = load_definitions(file_path)
        print(f"Loaded {len(word_data)} definitions")
        
        # Get word list for embeddings
        words = [data['word'] for data in word_data]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = get_embeddings(words)
        
        # Upsert to Pinecone
        print("Upserting to Pinecone...")
        upsert_definitions(
            index,
            word_data,
            embeddings,
            namespace=None if namespace == "General" else namespace
        )
    
    print("\nAll definition files processed!")

def main():
    """Main function to process and upsert all definitions."""
    process_definitions_folder()

if __name__ == "__main__":
    main()

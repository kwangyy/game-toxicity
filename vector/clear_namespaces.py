import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from pathlib import Path

def get_namespace_names():
    # Get the definitions directory path
    definitions_dir = Path(__file__).parent / 'definitions'
    
    # Get all CSV files and extract their names without extension
    namespaces = [
        file.stem for file in definitions_dir.glob('*.csv')
    ]
    
    # Map "General" to empty string (default namespace)
    return ['' if ns == 'General' else ns for ns in namespaces]

def clear_specific_namespaces():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('PINECONE_API_KEY')
    index_name = "dsa4288"
    
    # Check if environment variables are set
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME not found in environment variables")
    
    print(f"Using index: {index_name}")
    
    # Initialize Pinecone with API version
    pc = Pinecone(
        api_key=api_key,
    )
    
    index = pc.Index(index_name)
    
    try:
        # Get namespaces from CSV files
        target_namespaces = get_namespace_names()
        print(f"Found the following namespaces to clear: {', '.join(ns if ns else 'General' for ns in target_namespaces)}")
        
        # Get existing namespaces from Pinecone
        stats = index.describe_index_stats()
        existing_namespaces = stats.get('namespaces', {}).keys()
        
        # Clear each namespace
        for namespace in target_namespaces:
            display_name = 'General' if namespace == '' else namespace
            if namespace in existing_namespaces or namespace == '':
                print(f"Clearing namespace: {display_name}")
                # For default namespace, don't specify namespace parameter
                if namespace == '':
                    index.delete(delete_all=True)
                else:
                    index.delete(delete_all=True, namespace=namespace)
                print(f"Successfully cleared namespace: {display_name}")
            else:
                print(f"Namespace '{display_name}' not found in Pinecone index - skipping")
            
        print("Finished clearing specified namespaces!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        clear_specific_namespaces()
    except ValueError as e:
        print(f"Configuration Error: {str(e)}")
        print("Please make sure both PINECONE_API_KEY and PINECONE_INDEX_NAME are set in your .env file")
    except Exception as e:
        print(f"Unexpected error: {str(e)}") 
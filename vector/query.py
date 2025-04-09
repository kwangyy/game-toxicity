import os
import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
from Levenshtein import distance
import pandas as pd

# Load environment variables
load_dotenv()

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

def initialize_pinecone() -> Pinecone:
    """Initialize Pinecone connection."""
    api_key = os.getenv("PINECONE_API_KEY")
    
    if not api_key:
        raise ValueError("PINECONE_API_KEY must be set in .env")
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key)
    return pc.Index("dsa4288")

def normalize_edit_distance(edit_dist: int, max_length: int) -> float:
    """Normalize edit distance to [0,1] range."""
    return min(edit_dist / max_length, 1.0)

def get_word_embeddings(words: List[str]) -> np.ndarray:
    """Get OpenAI embeddings for a list of words."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=words
    )
    return np.array([item.embedding for item in response.data])

def find_best_match(
    query_word: str,
    query_embedding: np.ndarray,
    namespace: str,
    index: Pinecone,
    alpha: float = 0.7,
    threshold: float = 0.7
) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """
    Find the best matching word in a namespace.
    
    Returns:
        Tuple of (matched word, definition, score) or (None, None, None) if no match above threshold
    """
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=5,
        include_metadata=True,
        namespace=namespace if namespace != "General" else None
    )
    
    best_word = None
    best_definition = None
    best_score = 0
    
    for match in results.matches:
        word = match.metadata['word']
        definition = match.metadata['definition']
        cosine_score = match.score
        
        # Calculate normalized edit distance
        edit_dist = distance(query_word.lower(), word.lower())
        max_length = max(len(query_word), len(word))
        normalized_edit = normalize_edit_distance(edit_dist, max_length)
        
        # Combine scores
        final_score = alpha * cosine_score + (1 - alpha) * (1 - normalized_edit)
        
        if final_score > best_score:
            best_score = final_score
            best_word = word
            best_definition = definition
    
    if best_score >= threshold:
        return best_word, best_definition, best_score
    return None, None, None

def match_keywords_to_definitions(
    keywords: List[str],
    game: str,
    region: Optional[str] = None,
    threshold: float = 0.7
) -> Dict[str, Dict[str, Any]]:
    """
    Match keywords to definitions from relevant namespaces.
    
    Args:
        keywords: List of keywords to find definitions for
        game: Game namespace to search in
        region: Optional region namespace to search in
        threshold: Minimum score threshold for matches
    
    Returns:
        Dictionary mapping keywords to their matched definitions and scores
    """
    # Initialize Pinecone
    index = initialize_pinecone()
    
    # Get embeddings for all keywords at once
    keyword_embeddings = get_word_embeddings(keywords)
    
    # Set up namespaces
    namespaces = ["General", game]
    if region:
        namespaces.append(region)
    
    # Match each keyword
    results = {}
    for keyword, embedding in zip(keywords, keyword_embeddings):
        keyword_matches = {}
        
        # Try each namespace in order (specific to general)
        for namespace in reversed(namespaces):
            matched_word, definition, score = find_best_match(
                keyword,
                embedding,
                namespace,
                index,
                threshold=threshold
            )
            
            if matched_word:
                keyword_matches[namespace] = {
                    'matched_word': matched_word,
                    'definition': definition,
                    'score': score
                }
        
        if keyword_matches:
            results[keyword] = keyword_matches
    
    return results

def main():
    """Main function to demonstrate keyword matching."""
    # Example usage
    keywords = ["toxik", "feeding", "lag", "nguyens"]
    game = "League of Legends"
    region = "Southeast Asia"
    
    print(f"\nMatching keywords: {keywords}")
    print(f"Game: {game}")
    print(f"Region: {region}")
    
    matches = match_keywords_to_definitions(keywords, game, region)
    
    print("\nResults:")
    print("-" * 80)
    for keyword, namespace_matches in matches.items():
        print(f"\nKeyword: {keyword}")
        for namespace, match_info in namespace_matches.items():
            print(f"\n  {namespace} namespace:")
            print(f"    Matched word: {match_info['matched_word']}")
            print(f"    Definition: {match_info['definition']}")
            print(f"    Score: {match_info['score']:.4f}")

if __name__ == "__main__":
    main()
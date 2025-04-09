# DSA4288
A toxicity detection system for gaming chat messages using RAG (Retrieval Augmented Generation) with Pinecone vector database and LLM-based analysis.

## Project Structure

### /vector
Contains the core vector database operations using Pinecone. Handles initialization, upserting, querying, and namespace management for the vector store. This directory manages the RAG component of the system, allowing for efficient retrieval of gaming-related definitions and context.

### /process
Houses data processing utilities for preparing chat messages. Currently includes functionality to process JSON chat data into pandas DataFrames, serving as the data preparation pipeline for the toxicity detection system.

### /experiments
Contains multiple experimental approaches for toxicity detection, from basic prediction to complete workflows with gaming context enhancement. Includes visualization tools and comparison metrics, with results showing improved accuracy (F1-score of 0.74 for toxic messages) when using the complete workflow with gaming definitions.

### /inference
Contains the core inference logic for keyword extraction and toxicity prediction. Implements the main prediction pipeline using LLM-based analysis, with utilities for batch processing and prompt management.

### /prompts
Stores prompt templates and utility functions for generating context-aware prompts. Includes specialized prompts for keyword extraction and toxicity prediction, with support for gaming-specific terminology.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DSA4288.git
cd DSA4288
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with:
```env
PINECONE_API_KEY=your_pinecone_key
HF_TOKEN=your_huggingface_token
```
# Gaming Toxicity Detection Experiments

This folder contains experiments for testing different approaches to toxicity detection and message rewriting for gaming text.

## Available Experiments

1. **Base Prediction (`1_base_prediction.py`)**: 
   - Simple toxicity prediction on original messages
   - No context enhancement or rewriting
   - Quickest to run but potentially less accurate

2. **Full Workflow without Definitions (`2_workflow_without_definitions.py`)**:
   - Complete workflow with keyword extraction and rewriting
   - Does not use gaming terminology definitions
   - Moderate accuracy improvement over base prediction

3. **Complete Workflow with Definitions (`3_complete_workflow.py`)**:
   - Full workflow including keyword extraction, definitions, rewriting, and prediction
   - Incorporates gaming terminology definitions from the `prompts.utils` module
   - Most advanced approach with the highest potential accuracy

4. **Combined Runner (`run_all.py`)**:
   - Runs all three experiments in sequence
   - Generates a comparison file with results from all approaches
   - Calculates accuracy metrics if ground truth is available

## Running the Experiments

### Prerequisites

- Make sure `process/processed_data.csv` exists with the following columns:
  - `message`: The original message to analyze
  - `game`: The game the message is from
  - `location`: Geographic location/region of the message (optional)
  - `toxicity`: Boolean ground truth for toxicity (optional, used for accuracy calculation)

### Running Individual Experiments

```bash
# Run base prediction only
python -m experiments.1_base_prediction

# Run full workflow without definitions
python -m experiments.2_workflow_without_definitions

# Run complete workflow with definitions
python -m experiments.3_complete_workflow
```

### Running All Experiments

```bash
# Run all experiments and generate comparison
python -m experiments.run_all
```

## Results

All results will be saved in the `experiments/results/` directory:

- `base_prediction_results.csv`: Results from experiment 1
- `workflow_without_definitions_results.csv`: Results from experiment 2
- `complete_workflow_results.csv`: Results from experiment 3
- `experiment_comparison.csv`: Side-by-side comparison of all experiments (when using `run_all.py`)

## Gaming Terminology Definitions

The complete workflow integrates with the `prompts.utils.retrieve_definitions()` function, which is designed to be a placeholder for a future RAG (Retrieval-Augmented Generation) system. 

Currently, this function:
- Provides mock definitions based on the game and keywords
- Includes region-specific gaming terminology based on location
- Uses a predefined dictionary of common gaming terms
- Returns definitions tailored to the specific game context and extracted keywords

When implementing a real RAG system, you'll only need to update the `retrieve_definitions()` function in `prompts/utils.py` without changing the experiment code.

## Customization

The experiments can be customized in several ways:

- **Model**: All experiments use Llama 3.1 8B by default, but you can specify a different model
- **Batch Size**: Adjust the limit parameter to process only a subset of messages (useful for testing)
- **Definitions**: Enhance or modify the definitions in `prompts/utils.py` to test impact on results

For more advanced customization, see the documentation in each script. 
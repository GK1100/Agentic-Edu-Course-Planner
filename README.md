# Agentic Course Planner

An AI-powered academic advising assistant designed to parse course catalogs and provide grounded, hallucination-resistant responses.

## Prerequisites
- Python 3.10+
- Conda (Miniconda/Anaconda)
- Ollama running locally containing the Mistral model: `mistral-large-3:675b-cloud`

## Environment Setup
Run these commands in your Powershell or CMD terminal to run the UI application!

```bash
# Activate the correct environment
conda activate purple

# Launch the Agentic Assistant Wait for "Running on local URL: http://0.0.0.0:7860"
python app.py
```

## Running Evaluation Suite
If you want to run the 25-question evaluation suite to check the Verifier's behavior across multiple edge cases (Missing grades, multi-hop prerequisites, logical fabrication testing), simply run:
```bash
conda activate purple
python evaluation.py
```

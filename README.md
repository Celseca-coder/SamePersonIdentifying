# Person Re-Identification with LMMs

This project implements a Person Re-Identification (ReID) evaluation framework using Large Multimodal Models (LMMs) like GPT-4o or Qwen via API.

## Project Structure

- `src/data_loader.py`: Handles loading the Market-1501 dataset.
- `src/reid_agent.py`: Contains agent implementations (Mock, OpenAI, LangChain, Random).
- `src/run_evaluation.py`: Main script to run the evaluation loop and generate reports.
- `src/report.md`: Generated report file.

## Requirements

- Python 3.8+
- Dependencies: `openai`, `langchain-openai`, `pillow`, `matplotlib` (optional)

```bash
pip install openai langchain-openai pillow
```

## Usage

### 1. Mock Evaluation (Test Logic)
Running with `mock` agent checks filename IDs to verify the data loader and evaluation pipeline correct.

```bash
python src/run_evaluation.py --agent mock --trials 10
```

### 2. OpenAI Evaluation
Run with GPT-4o (or other OpenAI compatible models). Requires `OPENAI_API_KEY`.

```bash
python src/run_evaluation.py --agent openai --api_key YOUR_KEY_HERE --trials 20 --gallery_size 10
```

### 3. LangChain Evaluation
Run via LangChain interface (useful for connecting to Qwen or other providers via LangChain integrations).

```bash
python src/run_evaluation.py --agent langchain --api_key YOUR_KEY_HERE
```

## Dataset
This code is configured to use `l:\CV\data\1\Market-1501-v15.09.15` by default. You can change this with `--data_dir`.

## Multimodal Evaluation
The current implementation focuses on Image-to-Image ReID evaluating Rank-1 accuracy. To simulate multimodal or cross-camera evaluation, simply run the evaluation as the dataset loader automatically samples positive matches from different cameras/sequences.

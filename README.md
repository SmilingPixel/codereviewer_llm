# CodeReviewer-LLM

CodeReviewer-LLM is a code refinement and review system based on Large Language Models (LLMs). It is designed to take code snippets and review comments as input and generate improved code according to the review. The system supports training with LoRA (Low-Rank Adaptation), inference, and evaluation using BLEU and Exact Match metrics.

## Features

- **LoRA-based fine-tuning** for efficient model adaptation.
- **Custom data processing** for code review and refinement tasks.
- **Evaluation scripts** for BLEU and Exact Match metrics.
- **Configurable training and testing** via JSON config files.

## Project Structure

```
codereviewer_llm/
├── calc_metrics.py      # Evaluation script for BLEU and EM
├── config.py           # Configuration classes
├── data.py             # Data loading and processing
├── main.py             # Entry point for training/testing
├── test.py             # Inference and testing
├── train.py            # Training script
├── README.md           # This file
└── ...                 # Other files and datasets
```

## Setup

1. **Clone the repository** and install dependencies:
    ```bash
    git clone <repo-url>
    cd codereviewer_llm
    uv sync
    ```

2. **Prepare configuration files** (`global_config.json`, `train_config.json`, `test_config.json`) in the project root. Example:
    ```json
    {
      "model_id": "your-model-id",
      "dataset_path": "datasets/code_refinement/train.jsonl",
      "output_dir": "output"
    }
    ```

3. **Prepare your dataset** in JSONL format with fields such as `old`, `new`, `comment`, etc.

## Training

To train the model with LoRA fine-tuning:

```bash
uv run main.py
```

Ensure `do_train` is set to `true` in `global_config.json`.

## Testing / Inference

To run inference and generate model outputs:

```bash
uv run main.py
```

Ensure `do_test` is set to `true` in `global_config.json`.

## Evaluation

To evaluate model outputs using BLEU and Exact Match:

```bash
uv run calc_metrics.py \
  --ref_file path/to/ref-test.jsonl \
  --model_output_file path/to/test_results.jsonl \
  --results_output_file evaluation_results.json
```

## Notes

- The project uses HuggingFace Transformers and PEFT for model handling and LoRA.
- For best results, ensure your GPU supports bfloat16 or adjust `torch_dtype` as needed.
- The system prompt and chat template are hardcoded for code review tasks.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)

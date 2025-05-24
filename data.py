from datasets import Dataset
import pandas as pd
from transformers.tokenization_utils import PreTrainedTokenizer

def load_dataset(path: str) -> Dataset:
    df = pd.read_json(path)
    ds = Dataset.from_pandas(df)
    return ds

def process_example(example: dict[str, str], tokenizer) -> dict[str, list[int]]:
    # Hardcoded system prompt
    system_prompt = (
        "You are an AI code reviewer. Given a code snippet and a review comment, "
        "output the improved code according to the review. Do not add any other comments or explanations."
    )
    MAX_LENGTH = 4096

    # Compose the user instruction
    instruction = (
        f"Old hunk:\n```\n{example['old_hunk']}\n```\n"
        f"Old code:\n```\n{example['oldf']}\n```\n"
        f"Review comment:\n```\n{example['comment']}\n```\n"
        "Please output the improved code.\n"
    )

    # Use the chat template
    instruction_tokens = tokenizer(
        f"<s><|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n",
        add_special_tokens=False
    )
    response_tokens = tokenizer(f"```\n{example['new']}\n```", add_special_tokens=False)

    input_ids = instruction_tokens["input_ids"] + response_tokens["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction_tokens["attention_mask"] + response_tokens["attention_mask"] + [1]
    labels = [-100] * len(instruction_tokens["input_ids"]) + response_tokens["input_ids"] + [tokenizer.pad_token_id]

    # Truncate if necessary
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def get_and_process_dataset(path: str, tokenizer: PreTrainedTokenizer) -> Dataset:
    dataset = load_dataset(path)
    processed_dataset = dataset.map(
        lambda x: process_example(x, tokenizer),
        remove_columns=dataset.column_names,
    )
    return processed_dataset
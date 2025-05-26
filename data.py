from datasets import Dataset
import pandas as pd
from transformers.tokenization_utils import PreTrainedTokenizer
from config import GlobalConfig
import torch


class CodeRefinementDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items["input_ids"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(GlobalConfig.device) for key, val in self.items.items()}
        return item

def load_dataset(path: str) -> Dataset:
    df = pd.read_json(path, lines=True)
    df = df.drop(columns=["ids"])
    ds = Dataset.from_pandas(df, preserve_index=False)
    return ds

def process_example(example: dict[str, str], tokenizer) -> dict[str, list[int]]:
    # Hardcoded system prompt
    system_prompt = (
        "You are an AI code reviewer. Given a code snippet and a review comment, "
        "output the improved code according to the review. Do not add any other comments or explanations."
    )
    MAX_LENGTH = 4096
    OLD_HUNK_MAX_LENGTH = 128
    HUNK_MAX_LENGTH = 128
    OLDF_MAX_LENGTH = 2048
    OLD_MAX_LENGTH = 512
    NEW_MAX_LENGTH = 512
    COMMENT_MAX_LENGTH = 512

    # Truncate the old hunk, oldf, and comment if they exceed their respective max lengths
    example["old_hunk"] = example["old_hunk"][:OLD_HUNK_MAX_LENGTH]
    example["hunk"] = example["hunk"][:HUNK_MAX_LENGTH]
    example["oldf"] = example["oldf"][:OLDF_MAX_LENGTH]
    example["old"] = example["old"][:OLD_MAX_LENGTH]
    example["new"] = example["new"][:NEW_MAX_LENGTH]
    example["comment"] = example["comment"][:COMMENT_MAX_LENGTH]

    # Compose the user instruction
    instruction = (
        f"Old hunk:\n```\n{example['old_hunk']}\n```\n"
        f"Old code:\n```\n{example['oldf']}\n```\n"
        f"old diff:\n```\n{example['old']}\n```\n"
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

    res = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

    return res


def get_and_process_dataset(path: str, tokenizer) -> Dataset:
    dataset = load_dataset(path)
    processed_dataset = dataset.map(
        lambda x: process_example(x, tokenizer),
        remove_columns=dataset.column_names,
    )
    # processed_dataset = CodeRefinementDataset(processed_dataset)
    return processed_dataset
    
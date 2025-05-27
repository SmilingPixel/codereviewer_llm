from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForCausalLM
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from data import process_example
from config import TestConfig
import torch

def test():
    model_id = TestConfig.model_id
    model = AutoPeftModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    # model = AutoModelForCausalLM.from_pretrained(model_id).to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    model.eval()

    example = {
        "old_hunk": "",
        "hunk": "",
        "oldf": "",
        "old": "",
        "new": "",
        "comment": ""
    }

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

    instruction = (
        f"Old hunk:\n```\n{example['old_hunk']}\n```\n"
        f"Old code:\n```\n{example['oldf']}\n```\n"
        f"old diff:\n```\n{example['old']}\n```\n"
        f"Review comment:\n```\n{example['comment']}\n```\n"
        "Please output the improved code.\n"
    )

    messages = [
        {"role": "system", "content": "You are an AI code reviewer. Given a code snippet and a review comment, output the improved code according to the review. Do not add any other comments or explanations."},
        {"role": "user", "content": instruction}
    ]

    inputs = tokenizer.apply_chat_template(messages,
                                        add_generation_prompt=True,
                                        tokenize=True,
                                        return_tensors="pt",
                                        return_dict=True,
                                        enable_thinking=False).to('cuda')

    gen_kwargs = {"max_length": 4096, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

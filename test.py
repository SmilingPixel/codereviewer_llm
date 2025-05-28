from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForCausalLM
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from data import get_instruction_and_response, load_dataset
from config import TestConfig
import torch
import json

def test():
    model_id = TestConfig.model_id
    model = AutoPeftModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, padding_side="left")
    model.eval()

    dataset = load_dataset(TestConfig.dataset_path)

    inst_with_resp = dataset.map(
        get_instruction_and_response,
        remove_columns=dataset.column_names,
        num_proc=8
    )

    result = inst_with_resp.map(
        lambda x: infer_example_batch(x, model, tokenizer),
        batched=True,
        batch_size=16,
        remove_columns=inst_with_resp.column_names,
    )
    # result = inst_with_resp.map(
    #     lambda x: infer_example(x, model, tokenizer),
    #     remove_columns=inst_with_resp.column_names
    # )
    result.to_json(f"{TestConfig.output_dir}/test_results.jsonl", orient="records", lines=True)

def infer_example_batch(batch, model, tokenizer):
    chats = [[
        {"role": "system", "content": "You are an AI code reviewer. Given a code snippet and a review comment, output the improved code according to the review. Do not add any other comments or explanations."},
        {"role": "user", "content": instruction}
    ] for instruction in batch["instruction"]]

    inputs = [tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            #return_dict=True,
            enable_thinking=False
        ) for messages in chats]
    inputs = tokenizer(inputs, return_tensors="pt", padding=True).to("cuda")

    gen_kwargs = {"max_length": 4096, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        input_length = inputs['input_ids'].shape[1]
        outputs = outputs[:, input_length:]
    return {
        "output": tokenizer.batch_decode(outputs, skip_special_tokens=True)
    }


def infer_example(example, model, tokenizer):
    messages = [
            {"role": "system", "content": "You are an AI code reviewer. Given a code snippet and a review comment, output the improved code according to the review. Do not add any other comments or explanations."},
            {"role": "user", "content": example["instruction"]}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False
    ).to('cuda')

    gen_kwargs = {"max_length": 4096, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {
        "output": decoded_output
    }

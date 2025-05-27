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

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model.eval()

    dataset = load_dataset(TestConfig.dataset_path)

    results = []

    for example in dataset:
        instruction, resp = get_instruction_and_response(example)
        messages = [
            {"role": "system", "content": "You are an AI code reviewer. Given a code snippet and a review comment, output the improved code according to the review. Do not add any other comments or explanations."},
            {"role": "user", "content": instruction}
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
            results.append(decoded_output)

    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

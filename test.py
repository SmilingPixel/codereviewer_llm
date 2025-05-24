from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForCausalLM
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
import torch

def test(model_id: str, tokenizer):
    # Load base model and tokenizer
    # model = AutoPeftModelForCausalLM.from_pretrained(trained_model_path).to('cuda')
    model = AutoModelForCausalLM.from_pretrained(model_id).to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    model.eval()

    # Example input
    prompt = "我是来自上海杨浦的周迅，你可以通过 xunzhou24@m.fudan.edu.cn 联系我，我想问一下，AMD AI 9 HX 370 的性能如何？"

    messages = [
        {"role": "system", "content": "将文本中的name、address、email、question提取出来，以json格式输出，字段为name、address、email、question，值为文本中提取出来的内容。"},
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(messages,
                                        add_generation_prompt=True,
                                        tokenize=True,
                                        return_tensors="pt",
                                        return_dict=True,
                                        enable_thinking=False).to('cuda')

    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
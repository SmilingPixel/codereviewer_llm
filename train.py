from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from datasets import Dataset
import torch
from peft import LoraConfig, TaskType, get_peft_model
from .data import get_and_process_dataset
from .config import TrainingConfig
from transformers.models.auto.tokenization_auto import AutoTokenizer

def train_lora_sft():
    model_id = TrainingConfig.model_id
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False, # 训练模式
        r=8, # Lora 秩
        lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1 # Dropout 比例
    )
    model = get_peft_model(model, config)

    args = TrainingArguments(
        output_dir="output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=1,
        num_train_epochs=3,
        save_steps=50,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to=["wandb"],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    train_ds = get_and_process_dataset(TrainingConfig.dataset_path, tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()
    trainer.save_model(TrainingConfig.output_dir)

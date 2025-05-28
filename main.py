from datasets import Dataset
from datetime import datetime
import pandas as pd
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForCausalLM
from config import GlobalConfig, TrainingConfig, TestConfig
import torch

if __name__ == "__main__":

    # Load config from files if available
    GlobalConfig.load_from_file("global_config.json")
    TrainingConfig.load_from_file("train_config.json")
    TestConfig.load_from_file("test_config.json")


    TrainingConfig.output_dir = f"output_{datetime.now().strftime('%Y%m%d%H%M')}"    

    # import wandb
    # wandb.login()

    if GlobalConfig.do_train:
        from train import train_lora_sft
        train_lora_sft()
    
    if GlobalConfig.do_test:
        from test import test
        test()

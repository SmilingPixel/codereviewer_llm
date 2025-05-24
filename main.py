from datasets import Dataset
import pandas as pd
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForCausalLM
from .config import GlobalConfig, TrainingConfig
from .train import train_lora_sft
import torch

if __name__ == "__main__":

    GlobalConfig.do_train = True
    GlobalConfig.do_test = False


    TrainingConfig.model_id = 'THUDM/chatglm-6b'
    TrainingConfig.output_dir = "output"
    TrainingConfig.dataset_path = GlobalConfig.dataset_path
    

    import wandb
    wandb.login()

    if GlobalConfig.do_train:
        train_lora_sft()
    
    if GlobalConfig.do_test:
        pass

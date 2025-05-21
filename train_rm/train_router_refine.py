import os

os.environ["HF_HOME"] = [REDACTED]
os.environ["TRANSFORMERS_CACHE"] = [REDACTED]
os.environ["HF_DATASETS_CACHE"] = [REDACTED]
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import json
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset,Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy

import wandb
from peft import get_peft_model, LoraConfig, TaskType
# =================== Args =================== #
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--train_set_path', type=str, default=[REDACTED])
    parser.add_argument('--output_path', type=str, default=[REDACTED])
    parser.add_argument('--per_device_train_batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--loss_type', type=str, choices=['mse', 'smooth_l1', 'bt','ce','bce'], default='bt')
    parser.add_argument('--max_length', type=int, default=131072)
    parser.add_argument('--save_every_steps', type=int, default=1000)
    parser.add_argument('--eval_every_steps', type=int, default=1000)
    parser.add_argument('--bf16', action='store_true')
    return parser.parse_args()


# =================== Model =================== #
class RegressionRewardModel(nn.Module):
    def __init__(self, model_name, output_dim=4, freeze_backbone=True):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
        )


        # Apply LoRA to the backbone
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # Commonly used target modules; adjust if needed
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        self.backbone.print_trainable_parameters()


        hidden_size = self.backbone.config.hidden_size
        self.regression_head = nn.Linear(hidden_size, output_dim)


        # if self.backbone.model.embed_tokens.weight.dtype == torch.bfloat16:
        #     self.regression_head = self.regression_head.to(dtype=torch.bfloat16)

        try:
            dtype = self.backbone.model.embed_tokens.weight.dtype
        except AttributeError:
            dtype = next(self.backbone.parameters()).dtype
        self.regression_head = self.regression_head.to(dtype=dtype)
        

        # Only LoRA blocks + regression head are trainable
        for param in self.backbone.parameters():
            param.requires_grad = param.requires_grad or param.__class__.__name__.startswith("Lora")

        # if freeze_backbone:
        #     for param in self.backbone.parameters():
        #         param.requires_grad = False


        self.to(torch.device("cuda"))

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True,output_hidden_states=True)
        

        # outputs.hidden_states will be a tuple of hidden states from all layers.
        # The last element is the hidden state of the last layer.
        hidden_states = outputs.hidden_states[-1] # Modify this line
        #hidden_states = outputs.last_hidden_state

        lengths = attention_mask.sum(dim=1) - 1
        last_token_hidden_states = hidden_states[torch.arange(hidden_states.size(0)), lengths]
        scores = self.regression_head(last_token_hidden_states)
        return scores


# =================== Dataset =================== #
def build_dataset(tokenizer, path):
    document_types = ['triples', 'atomic facts', 'summaries', 'chunks']

[REDACTED]
        messages = [
            {'role': 'system', 'content': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'},
[REDACTED]
        ]

        conv_formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

        tokenized = tokenizer(conv_formatted, truncation=True, max_length=tokenizer.model_max_length)

        label_vector = [0.0] * 4
[REDACTED]
[REDACTED]
            label_vector[idx] = 1.0

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': label_vector,
        }

    # ===== LOAD JSON FILE ===== #
    with open(path, 'r') as f:
        raw_data = json.load(f)  # raw_data is a list of dicts

    # Turn it into a Dataset object
    ds = Dataset.from_list(raw_data)

    # Tokenize
    ds = ds.map(tokenize, num_proc=8)

    return ds


# =================== Collator =================== #
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_features = [{'input_ids': f['input_ids'], 'attention_mask': f['attention_mask']} for f in features]
        labels = torch.tensor([f['labels'] for f in features], dtype=torch.float)

        batch = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch['labels'] = labels
        return batch


# =================== Trainer =================== #

def bradley_terry_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    scores: Tensor of shape [batch_size, 4]
    labels: One-hot tensor of shape [batch_size, 4] where only the best method is 1
    """
    batch_size, num_methods = scores.size()
    loss = 0.0
    for i in range(batch_size):
        s = scores[i]
        g = torch.argmax(labels[i])  # index of the best method
        for j in range(num_methods):
            if j == g:
                continue
            loss += -torch.log(torch.sigmoid(s[g] - s[j]) + 1e-8)  # Add epsilon for stability
    return loss / (batch_size * (num_methods - 1))


class RewardTrainer(Trainer):

    def __init__(self, regression_save_path=None,loss_type='mse', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.regression_save_path = regression_save_path
        self._saved_epochs = set()  # Track which epochs have been saved

    

    def compute_loss(self, model, inputs, return_outputs=False):
        rewards = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        labels = inputs['labels']

        if rewards.dtype != labels.dtype:
            labels = labels.to(dtype=rewards.dtype)


        if self.loss_type == 'mse':
            loss = nn.functional.mse_loss(rewards, labels)
        elif self.loss_type == 'smooth_l1':
            loss = nn.functional.smooth_l1_loss(rewards, labels)
        elif self.loss_type == 'bt':
            loss = bradley_terry_loss(rewards, labels)
        elif self.loss_type == 'ce':
            # Cross Entropy expects class indices as target, not one-hot vectors
            target_indices = torch.argmax(labels, dim=1)
            loss = nn.functional.cross_entropy(rewards, target_indices)
        elif self.loss_type == 'bce':
            # Binary Cross-Entropy with logits (multi-label)
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(rewards, labels)

        else:
            raise ValueError(f"Unknown loss type {self.loss_type}")

        if return_outputs:
            return loss, {"rewards": rewards}
        return loss

    # def on_epoch_end(self):
    #     if self.regression_save_path is not None:
    #         save_path = os.path.join(self.regression_save_path, f"regression_head_epoch_{self.state.epoch:.0f}.pt")
    #         torch.save(self.model.regression_head.state_dict(), save_path)
    #         print(f"Saved regression head at {save_path}")
            
    # def train(self, *args, **kwargs):
    #     output = super().train(*args, **kwargs)
    #     # After training completes, also save one final regression head
    #     if self.regression_save_path is not None:
    #         save_path = os.path.join(self.regression_save_path, "regression_head_final.pt")
    #         torch.save(self.model.regression_head.state_dict(), save_path)
    #         print(f"Saved final regression head at {save_path}")
    #     return output

    def training_step(self, *args, **kwargs):
        output = super().training_step(*args, **kwargs)

        # Check if we finished an epoch
        epoch = int(self.state.epoch)
        epoch_list = [i+1 for i in range(0, int(self.args.num_train_epochs))]
        if epoch not in self._saved_epochs and epoch in epoch_list:
            self._saved_epochs.add(epoch)
            self.save_regression_head(epoch)

        return output

    def save_regression_head(self, epoch):
        if self.regression_save_path is not None:

            # save_path = os.path.join(self.regression_save_path, f"regression_head_epoch_{epoch}.pt")
            # torch.save(self.model.regression_head.state_dict(), save_path)
            # print(f"Saved regression head at {save_path}")

            reg_path = os.path.join(self.regression_save_path, f"regression_head_epoch_{epoch}.pt")
            torch.save(self.model.regression_head.state_dict(), reg_path)
            print(f"Saved regression head at {reg_path}")

            # Save LoRA weights (adapter)
            if hasattr(self.model.backbone, "save_pretrained"):
                lora_path = os.path.join(self.regression_save_path, f"lora_epoch_{epoch}")
                self.model.backbone.save_pretrained(lora_path)
                print(f"Saved LoRA weights at {lora_path}")
# =================== Main =================== #

def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    os.environ["WANDB_API_KEY"] = "575cb450f8a80fa74005b5542d0f48625484ee6f"
    wandb.login()

    wandb.init(
        project="memory_router", 
        name=f"train-lora-{args.loss_type}-{os.path.basename(args.train_set_path)}",
        config=vars(args)
    )

    # Create regression save dir using train set name
    train_set_basename = os.path.splitext(os.path.basename(args.train_set_path))[0]
    regression_save_dir = os.path.join(args.output_path, train_set_basename)
    os.makedirs(regression_save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = args.max_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token



    train_dataset = build_dataset(tokenizer, args.train_set_path)

    model = RegressionRewardModel(model_name=args.model_name, output_dim=4)

    training_args = TrainingArguments(
        output_dir=args.output_path,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        bf16=args.bf16,
        evaluation_strategy="no",
        save_strategy="no",
        save_steps=args.save_every_steps,
        eval_steps=args.eval_every_steps,
        logging_steps=10,
        remove_unused_columns=False,
        report_to="wandb", 
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=args.max_length),
        loss_type=args.loss_type,
        regression_save_path=regression_save_dir
    )

    trainer.train()

    # Save final regression head
    torch.save(model.regression_head.state_dict(), os.path.join(regression_save_dir, "regression_head_final.pt"))

    # Save final LoRA adapter
    model.backbone.save_pretrained(os.path.join(regression_save_dir, "lora_final"))

    # Save the final model and tokenizer
    # trainer.save_model(args.output_path + "/final_model")
    # tokenizer.save_pretrained(args.output_path + "/final_model")

    # Save only the regression head
    #torch.save(model.regression_head.state_dict(), os.path.join(args.output_path, "regression_head.pt"))


if __name__ == "__main__":
    main()

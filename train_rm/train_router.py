########################
# Full Updated Reward Modeling Script
# - RegressionRewardModel (instead of AutoModelForSequenceClassification)
# - Supports MSE, SmoothL1, Regular BT, Margin BT, Scaled BT Losses
# - Keeps your original code commented for comparison
########################

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,  # original
    AutoModelForCausalLM,                 # new backbone
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy


# =================== Define Arguments =================== #
@dataclass
class ScriptArguments:
    local_rank: Optional[int] = field(default=-1)
    deepspeed: Optional[str] = field(default=None)
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=64)
    learning_rate: Optional[float] = field(default=2e-6)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B-Instruct")
    bf16: Optional[bool] = field(default=True)
    num_train_epochs: Optional[int] = field(default=1)
    train_set_path: Optional[str] = field(default="hendrydong/preference_700K")
    eval_set_path: Optional[str] = field(default="hendrydong/preference_700K")
    output_path: Optional[str] = field(default="./models/llama3_rm")
    gradient_checkpointing: Optional[bool] = field(default=True)
    optim: Optional[str] = field(default="paged_adamw_32bit")
    lr_scheduler_type: Optional[str] = field(default="cosine")
    max_length: Optional[int] = field(default=4096)
    save_every_steps: Optional[int] = field(default=999999)
    eval_every_steps: Optional[int] = field(default=999999)
    loss_type: Optional[str] = field(default='regular_bt')  # New arg

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


# =================== Tokenizer =================== #
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, use_fast=False)
# old
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id
# new
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.truncation_side = "left"
tokenizer.model_max_length = script_args.max_length


# =================== Dataset =================== #
def build_dataset(tokenizer, train_path, eval_path):
[REDACTED]
[REDACTED]
[REDACTED]
[REDACTED]
[REDACTED]
[REDACTED]
[REDACTED]
[REDACTED]
[REDACTED]
[REDACTED]
[REDACTED]
[REDACTED]

    ds = load_dataset(train_path, split="train").shuffle(seed=42)
    ds = ds.map(tokenize, num_proc=8)
    eval_dataset = ds.select(range(500))
    return ds, eval_dataset

train_dataset, eval_dataset = build_dataset(tokenizer, script_args.train_set_path, script_args.eval_set_path)
print("Training set:", len(train_dataset), " Eval set:", len(eval_dataset))







# =================== Model =================== #
class RegressionRewardModel(nn.Module):
    def __init__(self, model_name, output_dim=1, freeze_backbone=True):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        hidden_size = self.backbone.config.hidden_size
        self.regression_head = nn.Linear(hidden_size, output_dim)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        # === NEW ===
        # Get the "real" last token (after padding)
        # Find the lengths of each sequence
        attention_mask = attention_mask.to(hidden_states.device)
        lengths = attention_mask.sum(dim=1) - 1  # index of last non-padding token

        # Gather last token hidden states
        last_token_hidden_states = hidden_states[torch.arange(hidden_states.size(0)), lengths]  # (batch_size, hidden_dim)

        scores = self.regression_head(last_token_hidden_states)
        return scores

# old
# model = AutoModelForSequenceClassification.from_pretrained(...)
# new
model = RegressionRewardModel(
    model_name=script_args.model_name,
    output_dim=1,
    freeze_backbone=True
)


# =================== Loss Functions =================== #
def mse_loss(preds, targets):
    return nn.functional.mse_loss(preds, targets)

def smooth_l1_loss(preds, targets):
    return nn.functional.smooth_l1_loss(preds, targets)

def regular_bt_loss(pred1, pred2, label1, label2, margin=0.0):
    diff = (pred1 - pred2)
    label_diff = (label1 - label2)
    return torch.mean(torch.log(1 + torch.exp(-(diff * torch.sign(label_diff) - margin))))

def margin_bt_loss(pred1, pred2, label1, label2, margin=1.0):
    diff = (pred1 - pred2)
    label_diff = (label1 - label2)
    return torch.mean(torch.relu(margin - diff * torch.sign(label_diff)))

def scaled_bt_loss(pred1, pred2, label1, label2):
    diff = (pred1 - pred2)
    label_diff = (label1 - label2)
    scale = torch.abs(label_diff)
    return torch.mean(scale * torch.log(1 + torch.exp(-diff * torch.sign(label_diff))))


# =================== Data Collator =================== #
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        labels = []

        for feature in features:
            merged_features.append({"input_ids": feature["input_ids_j"], "attention_mask": feature["attention_mask_j"]})
            merged_features.append({"input_ids": feature["input_ids_k"], "attention_mask": feature["attention_mask_k"]})
            labels.append(torch.tensor([1.0]))  # positive
            labels.append(torch.tensor([0.0]))  # negative

        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = torch.stack(labels)
        return batch


# =================== Trainer =================== #
class RewardTrainer(Trainer):
    def __init__(self, *args, loss_type='regular_bt', **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type

    def compute_loss(self, model, inputs, return_outputs=False):
        rewards = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        if self.loss_type in ['mse', 'smooth_l1']:
            labels = inputs["labels"]
            loss_fn = mse_loss if self.loss_type == 'mse' else smooth_l1_loss
            loss = loss_fn(rewards, labels)
        else:
            bsz = rewards.size(0)
            assert bsz % 2 == 0, "Batch size must be even for pairwise comparison"
            jidx = torch.arange(0, bsz, 2)
            kidx = jidx + 1
            rewards_j = rewards[jidx]
            rewards_k = rewards[kidx]
            labels_j = inputs["labels"][jidx]
            labels_k = inputs["labels"][kidx]

            if self.loss_type == 'regular_bt':
                loss = regular_bt_loss(rewards_j, rewards_k, labels_j, labels_k)
            elif self.loss_type == 'margin_bt':
                loss = margin_bt_loss(rewards_j, rewards_k, labels_j, labels_k)
            elif self.loss_type == 'scaled_bt':
                loss = scaled_bt_loss(rewards_j, rewards_k, labels_j, labels_k)

        if return_outputs:
            return loss, {"rewards": rewards}
        return loss


def compute_metrics(eval_pred):
    return {}


# =================== Train =================== #
training_args = TrainingArguments(
    output_dir=script_args.output_path,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="steps",
    eval_steps=script_args.eval_every_steps,
    save_strategy="steps",
    save_steps=script_args.save_every_steps,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_ratio=0.03,
    report_to='wandb',
)

trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
    loss_type=script_args.loss_type,
)

trainer.train()

# Save final model
trainer.save_model(script_args.output_path + "/last_checkpoint")
tokenizer.save_pretrained(script_args.output_path + "/last_checkpoint")

######################### END #########################

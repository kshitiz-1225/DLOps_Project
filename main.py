from config import Config
from datasets import load_dataset
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from model import Wav2Vec2ForSpeechClassification, DataCollatorCTCWithPadding, compute_metrics

from transformers import AutoConfig, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Trainer, TrainingArguments


from typing import Any, Dict, Union, Optional, Tuple


import wandb

import torch
from torch.cuda.amp import autocast

from utils import *

import warnings
warnings.filterwarnings('ignore')

seed_everything(42)


args = Config()

wandb.init(project="Audio-Classification",
           name=f"{args.run_name}", tags=['wav2vec2', 'transformers'])

# wandb.config.update(args)


dataset_path = args.data_dir
save_path = args.save_path


data_files = {
    "train": f"{save_path}/train.csv",
    "validation": f"{save_path}/valid.csv",
    "test": f"{save_path}/test.csv",
}

dataset = load_dataset("csv", data_files=data_files)


train_dataset = dataset["train"]
eval_dataset = dataset["validation"]


input_column = "path"
output_column = "label"


label_list = train_dataset.unique(output_column)
label_list.sort()
num_labels = len(label_list)
print(f"A classification problem with {num_labels} classes:\n {label_list}")


model_name_or_path = "facebook/wav2vec2-base-100k-voxpopuli"
pooling_mode = "mean"


config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="   ",
)


setattr(config, 'pooling_mode', pooling_mode)


feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    model_name_or_path)
target_sampling_rate = feature_extractor.sampling_rate
print(f"The target sampling rate: {target_sampling_rate}")


def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(
        sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def preprocess_function(examples):
    speech_list = [speech_file_to_array_fn(
        path) for path in examples[input_column]]
    target_list = [label_to_id(label, label_list)
                   for label in examples[output_column]]

    result = feature_extractor(speech_list, sampling_rate=target_sampling_rate)
    result["labels"] = list(target_list)

    return result


train_dataset = train_dataset.map(
    preprocess_function,
    batch_size=100,
    batched=True,
)
eval_dataset = eval_dataset.map(
    preprocess_function,
    batch_size=100,
    batched=True,
)

data_collator = DataCollatorCTCWithPadding(
    feature_extractor=feature_extractor, padding=True)

model = Wav2Vec2ForSpeechClassification.from_pretrained(
    model_name_or_path,
    config=config,
)

model.freeze_feature_extractor()


class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        # elif self.deepspeed:
        #     self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


training_args = TrainingArguments(

    dataloader_drop_last=args.drop_last,
    dataloader_num_workers=args.num_workers,
    dataloader_pin_memory=args.pin_memory,

    disable_tqdm=False,

    do_train=True,
    do_eval=True,
    do_predict=True,

    eval_steps=args.eval_steps,
    evaluation_strategy="steps",

    fp16=args.fp16,

    learning_rate=args.lr,
    warmup_steps=args.warmup_steps,

    num_train_epochs=args.n_epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,


    optim=args.optimizer,


    output_dir=args.model_save_path,
    overwrite_output_dir=True,

    save_steps=args.save_steps,
    save_total_limit=2,
    seed=args.seed,


    report_to=['wandb'],
    logging_steps=args.logging_step,
    logging_strategy='steps'

)


trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=feature_extractor,
)


torch.cuda.empty_cache()
trainer.train()

wandb.finish()

import numpy as np
import librosa
import torchaudio
import torch

import torch
from transformers import Wav2Vec2FeatureExtractor
from model import Wav2Vec2ForSpeechClassification

from datasets import load_dataset, load_metric
from sklearn.metrics import classification_report, accuracy_score

from config import Config


import os

args = Config()

model_name_or_path = "saved_data/gtzan-music/checkpoint-7900"

test_dataset = load_dataset(
    "csv", data_files={"test": f"{args.save_path}/test.csv"})["test"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    model_name_or_path)
target_sr = feature_extractor.sampling_rate
model = Wav2Vec2ForSpeechClassification.from_pretrained(
    model_name_or_path).to(device)


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(np.asarray(
        speech_array), orig_sr=sampling_rate, target_sr=target_sr)

    batch["speech"] = speech_array
    return batch


def predict(batch):
    features = feature_extractor(
        batch["speech"], sampling_rate=target_sr, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch


test_dataset = test_dataset.map(speech_file_to_array_fn)
result = test_dataset.map(predict, batched=True, batch_size=8)

label_dict = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco',
              4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}
label2id = {v: k for k, v in label_dict.items()}

label_names = list(label_dict.values())

y_true = [label2id[name] for name in result["label"]]
y_pred = result["predicted"]

print(classification_report(y_true, y_pred, target_names=label_names))

print(f'Accuracy score is {accuracy_score(y_true,y_pred)*100:.2f}%')

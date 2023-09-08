import librosa
import torchaudio


import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor
from model import Wav2Vec2ForSpeechClassification


import os


model_name_or_path = "saved_data/gtzan-music/checkpoint-7900"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    model_name_or_path)
model = Wav2Vec2ForSpeechClassification.from_pretrained(
    model_name_or_path).to(device)


label_dict = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco',
              4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}
label_names = list(label_dict.values())


def predict(file_path):
    target_sr = feature_extractor.sampling_rate
    speech_array, sampling_rate = torchaudio.load(file_path)
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(np.asarray(
        speech_array), orig_sr=sampling_rate, target_sr=target_sr)
    features = feature_extractor(
        speech_array, sampling_rate=target_sr, return_tensors="pt", padding=True)
    input_values = features.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    genre = label_names[pred_ids[0]]

    return genre

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm

import torchaudio
import sklearn
from sklearn.model_selection import train_test_split


from config import Config


args = Config()

dataset_path = args.data_dir
save_path = args.save_path
data = []

os.makedirs(save_path, exist_ok=True)

ignorefiles = ['jazz.00054.wav']  # Corrupted file


for path in Path(f'{dataset_path}/genres_original').glob("**/*.wav"):
    pathsep = '/'

    label = str(path).split(pathsep)[-2]
    name = str(path).split(pathsep)[-1]

    if name in ignorefiles:
        continue

    data.append({
        "path": path,
        "label": label
    })

df = pd.DataFrame(data)
df = df.sample(frac=1)

print('label_count')
print(df.groupby("label").count()[["path"]])


train_df, rest_df = train_test_split(df, test_size=0.2,  stratify=df["label"])
valid_df, test_df = train_test_split(
    rest_df, test_size=0.5,  stratify=rest_df["label"])

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df.to_csv(f"{save_path}/train.csv", encoding="utf-8", index=False)
valid_df.to_csv(f"{save_path}/valid.csv", encoding="utf-8", index=False)
test_df.to_csv(f"{save_path}/test.csv", encoding="utf-8", index=False)

print(train_df.shape)
print(valid_df.shape)
print(test_df.shape)

# -*- coding: utf-8 -*-
import os
from datasets import load_from_disk
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

DATASET_PATH = "../dataset_full_train_contact1_val"  

ds_dict = load_from_disk(DATASET_PATH)   
train_ds = ds_dict["train"]
val_ds   = ds_dict["val"]
test_ds  = ds_dict["test"]

def extract_mfcc_df(dataset, n_mfcc=13):
    records = []
    for sample in dataset:
        y  = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        # 13-dimension MFCC，shape=(n_mfcc, T)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std  = mfcc.std(axis=1)
        feat = np.concatenate([mfcc_mean, mfcc_std])
        lbl  = int(sample["label"])  # True→1, False→0
        records.append(np.concatenate([feat, [lbl]]))
    cols = [f"mfcc_mean_{i}" for i in range(1, n_mfcc+1)] + \
           [f"mfcc_std_{i}"  for i in range(1, n_mfcc+1)] + \
           ["label"]
    return pd.DataFrame(records, columns=cols)

df_train = extract_mfcc_df(train_ds)
df_val   = extract_mfcc_df(val_ds)
df_test  = extract_mfcc_df(test_ds)

df_train.to_excel("train_features.xlsx", index=False)
df_val.to_excel("val_features.xlsx",   index=False)
df_test.to_excel("test_features.xlsx", index=False)
print("Generated train/val/test_features.xlsx")

X_train = df_train.drop("label", axis=1)
y_train = df_train["label"]
X_test  = df_test.drop("label", axis=1)
y_test  = df_test["label"]

clf = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import tensorflow as tf
from datasets import load_from_disk
import librosa
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Masking, Bidirectional, LSTM, BatchNormalization,
    Dropout, Dense
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from sklearn.metrics import accuracy_score, classification_report

def extract_mfcc_deltas(dataset, n_mfcc=13):
    seqs, labels = [], []
    for sample in tqdm(dataset, desc="Extract MFCC+Δ"):
        y, sr = sample["audio"]["array"], sample["audio"]["sampling_rate"]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        #delta1 = librosa.feature.delta(mfcc, width=3)
        #delta2 = librosa.feature.delta(mfcc, width=3, order=2)
        #feats = np.vstack([mfcc, delta1, delta2]).T  # shape=(T, 3*n_mfcc)
        feats = np.vstack([mfcc]).T 
        seqs.append(feats)
        labels.append(int(sample["label"]))
    return seqs, np.array(labels)

DATASET_PATH = "./dataset_full_train_contact1_val_seg"
ds = load_from_disk(DATASET_PATH)
train_ds, val_ds, test_ds = ds["train"], ds["val"], ds["test"]

train_seqs, y_train = extract_mfcc_deltas(train_ds)
val_seqs,   y_val   = extract_mfcc_deltas(val_ds)
test_seqs,  y_test  = extract_mfcc_deltas(test_ds)

max_len = max(len(s) for s in train_seqs)
n_feats = train_seqs[0].shape[1]  # = 3 * n_mfcc
X_train = pad_sequences(train_seqs, maxlen=max_len, dtype='float32',
                        padding='post', value=0.0)
X_val   = pad_sequences(val_seqs,   maxlen=max_len, dtype='float32',
                        padding='post', value=0.0)
X_test  = pad_sequences(test_seqs,  maxlen=max_len, dtype='float32',
                        padding='post', value=0.0)

model = Sequential([
    Input(shape=(max_len, n_feats)),
    Masking(mask_value=0.0),
    Bidirectional(LSTM(64, return_sequences=True)),
    BatchNormalization(),
    Bidirectional(LSTM(32)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=['accuracy']
)
model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
    ModelCheckpoint('best_mfcc_lstm.h5', monitor='val_loss', save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=3,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

y_prob = model.predict(X_test, batch_size=32).ravel()
y_pred = (y_prob > 0.5).astype(int)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))

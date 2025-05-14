from datasets import load_from_disk
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import evaluate
import numpy as np
import torch

# set the seed
torch.manual_seed(0)
np.random.seed(0)

import uuid
# generate a unique identifier for the run
run_id = str(uuid.uuid4())
# set the run name
run_name = f"wav2vec2-base-cough-full-train-contact1-val-{run_id}-seg-100"

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
    )
    return inputs


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)


feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

cough_dataset = load_from_disk("/n/holylabs/LABS/protopapas_lab/Lab/Capstone_6/dataset_full_train_contact1_val_seg")

encoded = cough_dataset.map(preprocess_function, remove_columns="audio", batched=True)

from datasets import Features, Value
feature_names = list(encoded["train"].features.keys())
features_dict = {}

for name in feature_names:
    if name == "label":
        features_dict[name] = Value("int64")
    else:
        features_dict[name] = encoded["train"].features[name]

# Cast each split to the new feature types
for split in encoded.keys():
    encoded[split] = encoded[split].cast(Features(features_dict))


accuracy = evaluate.load("accuracy")


model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=2,
)

training_args = TrainingArguments(
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit = 2,
    learning_rate=3e-5,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=64,
    num_train_epochs=100,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    bf16=True,
    output_dir=f"/n/holylabs/LABS/protopapas_lab/Lab/Capstone_6/results/wav2vec2-base-cough-full-train-contact1-val-{run_name}-seg-100-epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["val"],
    processing_class=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()

# get best model and evaluate on test set
print(trainer.evaluate(encoded["test"]))

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

cough_dataset = load_from_disk("/n/holylabs/LABS/protopapas_lab/Lab/Capstone_6/coughdata/dataset")

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
    # "facebook/wav2vec2-base-960h",
    # "facebook/wav2vec2-large-960h-lv60-self",
    num_labels=2,  #attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
)

training_args = TrainingArguments(
    output_dir="results/wav2vec2-base-cough",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=64,
    num_train_epochs=30,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    bf16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["val"],
    processing_class=feature_extractor,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

# get best model and evaluate on test set
print(trainer.evaluate(encoded["test"]))

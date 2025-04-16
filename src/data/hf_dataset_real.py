from datasets import load_dataset, Audio
import datasets
import os
import json

# datasets.disable_caching()

MAX_DURATION_IN_SECONDS = 20.0

with open('/n/holylabs/LABS/protopapas_lab/Lab/Capstone_6/pneumonia_mapping.json') as f:
    # Load the JSON file
    XID_to_label = json.load(f)

# datasets.disable_caching()

MAX_DURATION_IN_SECONDS = 20.0

with open('./pneumonia_mapping.json') as f:
    # Load the JSON file
    XID_to_label = json.load(f)


def add_label_column(example):
    # Get the wav file path
    wav_path = example["path"]
    XID = wav_path.split(" (")[0].split("/")[-1]
    if XID in XID_to_label:
        label = XID_to_label[XID]
    else:
        # raise ValueError(f"XID {XID} not found in mapping file.")
        print(f"XID {XID} not found in mapping file. Assigning label 0.")
        label = False
    example["label"] = label
    return example


def add_length(example):
    example["input_length"] = (
        len(example["audio"]["array"]) / example["audio"]["sampling_rate"]
    )
    example["path"] = example["audio"]["path"]
    return example


def is_audio_length_in_range(input_length):
    return input_length < MAX_DURATION_IN_SECONDS


def map_hf_dataset(dataset, map_fn):
    for i, example in enumerate(dataset):
        example = map_fn(example)
    return dataset


# Load the dataset
dataset = load_dataset(
    "audiofolder",
    data_dir="/home/zinccat/project/coughdata/data",
    split="train",
    # keep_in_memory=True,
)

dataset = dataset.map(add_length)
dataset = dataset.filter(is_audio_length_in_range, input_columns=["input_length"])
dataset = dataset.map(add_label_column) #, keep_in_memory=True)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000)) #original as 48000

# split the dataset to train, val and test
dataset = dataset.train_test_split(test_size=0.2, seed=42)
# change the name of the split
dataset["val"], dataset["test"] = dataset["test"].train_test_split(0.5, seed=42).values()

print(dataset["train"][0])

# save the dataset
dataset.save_to_disk("/n/home07/shiji/Capstone_6/dataset")
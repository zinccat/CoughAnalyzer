from datasets import load_dataset, Audio
import datasets
import os

datasets.disable_caching()

MAX_DURATION_IN_SECONDS = 11.0


def add_label_column(example):
    # Get the wav file path
    wav_path = example["path"]
    # Replace the .wav extension with .txt to get the txt file path
    txt_path = wav_path.replace(".wav", ".txt")
    # Label is 1 if the txt file exists, otherwise 0
    example["label"] = 1 if os.path.exists(txt_path) else 0
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
    data_dir="CoughSegmentation/Data",
    split="train",
    keep_in_memory=True,
)

dataset = dataset.map(add_length, num_proc=6)
dataset = dataset.filter(is_audio_length_in_range, input_columns=["input_length"])
dataset = dataset.map(add_label_column, keep_in_memory=True)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# split the dataset to train, val and test
dataset = dataset.train_test_split(test_size=0.2, seed=42)
# change the name of the split
dataset["val"], dataset["test"] = dataset["test"].train_test_split(0.5, seed=42).values()

print(dataset["train"][0])

# save the dataset
dataset.save_to_disk("data/dataset_hf")

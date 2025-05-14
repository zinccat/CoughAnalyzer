from datasets import load_dataset, Audio
import datasets
import os
import json

# datasets.disable_caching()

MAX_DURATION_IN_SECONDS = 20.0

with open('./pneumonia_mapping.json') as f:
    # Load the JSON file
    XID_to_label = json.load(f)


def add_label_column(example):
    # Get the wav file path
    wav_path = example["path"]
    # wav_path = example["audio"]["path"]
    # print(wav_path)
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

with open('src/data/dataset_split_ids.json') as f:
    # Load the JSON file
    split_ids = json.load(f)

def get_split(split_name):
    # Get the list of XIDs for the given split
    xids = split_ids[split_name]
    # Create a filter function to check if the XID is in the list
    def filter_fn(wav_path):
        XID = wav_path["path"].split(" (")[0].split("/")[-1]
        return XID in xids
    return filter_fn

# Filter the dataset based on the split
def filter_dataset(dataset, split_name):
    filter_fn = get_split(split_name)
    return dataset.filter(filter_fn, input_columns=["audio"]) #, keep_in_memory=True)

# Load the dataset
dataset_train = load_dataset(
    "audiofolder",
    # data_dir="/home/zinccat/project/coughdata/data",
    data_dir="/home/zinccat/project/coughdata/seg/merged_audio_output",
    # data_dir="/home/zinccat/project/coughdata/first_contacts",
    # data_dir="/home/zinccat/project/coughdata/cough_mp3_full_yamnet_seg/merged",
    split="train",
    # keep_in_memory=True,
)
# only keep first 100 files
# dataset_train = dataset_train.select(range(100))
# Filter the dataset based on the split
dataset_train = filter_dataset(dataset_train, "train")


dataset_train = dataset_train.map(add_length)
dataset_train = dataset_train.filter(is_audio_length_in_range, input_columns=["input_length"])
dataset_train = dataset_train.map(add_label_column) #, keep_in_memory=True)
dataset_train = dataset_train.cast_column("audio", Audio(sampling_rate=16000)) #original as 48000

dataset = load_dataset(
    "audiofolder",
    # data_dir="/home/zinccat/project/coughdata/first_contacts",
    # data_dir="/home/zinccat/project/coughdata/first_contacts_yamnet_seg",
    data_dir="/home/zinccat/project/coughdata/first_contacts_seg",
    split="train",
)
dataset_val = filter_dataset(dataset, "val")
# dataset_val = dataset_val.select(range(100))
dataset_val = dataset_val.map(add_length)
dataset_val = dataset_val.filter(is_audio_length_in_range, input_columns=["input_length"])
dataset_val = dataset_val.map(add_label_column) #, keep_in_memory=True)
dataset_val = dataset_val.cast_column("audio", Audio(sampling_rate=16000)) #original as 48000

# dataset_val = filter_dataset(dataset_val, "val")
# dataset_test = filter_dataset(dataset_val, "test")

dataset_test = filter_dataset(dataset, "test")
# dataset_test = dataset_test.select(range(100))
dataset_test = dataset_test.map(add_length)
dataset_test = dataset_test.filter(is_audio_length_in_range, input_columns=["input_length"])
dataset_test = dataset_test.map(add_label_column) #, keep_in_memory=True)
dataset_test = dataset_test.cast_column("audio", Audio(sampling_rate=16000)) #original as 48000



dataset = datasets.DatasetDict({
    "train": dataset_train,
    "val": dataset_val,
    "test": dataset_test
})

# split the dataset to train, val and test
# dataset = dataset.train_test_split(test_size=0.2, seed=42)
# # change the name of the split
# dataset["val"], dataset["test"] = dataset["test"].train_test_split(0.5, seed=42).values()

print(dataset["train"][0])

print(f"Train: {len(dataset['train'])} XIDs")
print(f"Validation: {len(dataset['val'])} XIDs")
print(f"Test: {len(dataset['test'])} XIDs")

# save the dataset
# dataset.save_to_disk("/home/zinccat/project/coughdata/dataset_full_train_contact1_val")
# dataset.save_to_disk("/home/zinccat/project/coughdata/dataset_full_train_contact1_val_yamnet_seg")
dataset.save_to_disk("/home/zinccat/project/coughdata/dataset_full_train_contact1_val_seg")
# dataset.save_to_disk("/home/zinccat/project/coughdata/dataset_contact1")
# dataset.save_to_disk("/home/zinccat/project/coughdata/dataset_yamnet_full")
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from PIL import Image
from typing import Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def wav_to_image(
    wav_filename: str,
    label_filename: str,
    images_dir: str,
    label_dir: str,
    output_fileid: int,
    output_size: Tuple[int, int] = (640, 640),
    cough_class: int = 1,
):
    # Read the WAV file
    sample_rate, data = wavfile.read(wav_filename)

    # Create the time axis for plotting (limited to 11 seconds)
    if len(data) > 11 * sample_rate:
        # If the audio is longer than 11 seconds, remove it
        return
    audio_duration = len(data) / sample_rate
    scale = 11 / audio_duration
    # resizes the data to 11 seconds
    time = np.linspace(0, 11, num=len(data))

    # Plot the waveform
    plt.figure(figsize=(10, 10))

    # Plot the waveform with time on x-axis and amplitude on y-axis
    plt.plot(time, data[: len(time)], color="black")

    # Set fixed axis limits
    plt.xlim(0, 11)
    plt.ylim(-35000, 35000)

    # Remove the grid and legend
    plt.grid(False)
    plt.axis("off")

    # Save the plot to a file
    plt.savefig(
        Path(images_dir) / (str(output_fileid) + ".png"),
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close()

    # open fig and resize it to 640x640
    img = Image.open(Path(images_dir) / (str(output_fileid) + ".png")).convert("L")
    # resize the image
    img = img.resize(output_size)
    # save the image
    img.save(Path(images_dir) / (str(output_fileid) + ".png"))

    # use scale to resize the interval
    # first check if the txt file exists
    if os.path.exists(label_filename):
        with open(label_filename, "r") as f:
            lines = f.readlines()
            with open(Path(label_dir) / (str(output_fileid) + ".txt"), "w") as f2:
                for line in lines:
                    start, end = line.split()
                    start = int(float(start) / audio_duration * output_size[0])
                    end = int(float(end) / audio_duration * output_size[0])
                    assert start < end, f"start: {start}, end: {end}"
                    # if we want to keep start/end
                    # f2.write(
                    #     str(start)
                    #     + " "
                    #     + str(end)
                    #     + "\n"
                    # )

                    # if we want to keep the center of the interval
                    f2.write(
                        str(cough_class)
                        + " "
                        + str((start + end) // 2)  # x center
                        + " "
                        + "0"  # y center
                        + " "
                        + str(end - start)  # width
                        + " "
                        + "640"  # height
                        + "\n"
                    )

    else:
        with open(Path(label_dir) / (str(output_fileid) + ".txt"), "w") as f2:
            f2.write("")


if __name__ == "__main__":
    # wav_to_image(
    #     "CoughSegmentation/Data/0a03da19-eb19-4f51-9860-78ad95fa8cb5.wav",
    #     "CoughSegmentation/Data/0a03da19-eb19-4f51-9860-78ad95fa8cb5.txt",
    #     "data/sample/images",
    #     "data/sample/labels",
    #     -1,
    # )

    # read in all the files
    data_dir = "CoughSegmentation/Data"

    # filter out the files that are not wav files
    wav_files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]

    # split the files into train and test
    wav_files_train, wav_files_test = train_test_split(
        wav_files, test_size=0.2, random_state=42
    )
    # split test into test and validation
    wav_files_test, wav_files_val = train_test_split(
        wav_files_test, test_size=0.5, random_state=42
    )

    # create the directories
    images_dir = "data/images"
    label_dir = "data/labels"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # loop through the files and create the images
    images_dir_train = "data/images/train"
    label_dir_train = "data/labels/train"
    if not os.path.exists(images_dir_train):
        os.makedirs(images_dir_train)
    if not os.path.exists(label_dir_train):
        os.makedirs(label_dir_train)
    for i, wav_file in enumerate(tqdm(wav_files_train)):
        wav_to_image(
            os.path.join(data_dir, wav_file),
            os.path.join(data_dir, wav_file.replace(".wav", ".txt")),
            images_dir_train,
            label_dir_train,
            i + 1,
        )

    images_dir_test = "data/images/test"
    label_dir_test = "data/labels/test"
    if not os.path.exists(images_dir_test):
        os.makedirs(images_dir_test)
    if not os.path.exists(label_dir_test):
        os.makedirs(label_dir_test)
    for i, wav_file in enumerate(tqdm(wav_files_test)):
        wav_to_image(
            os.path.join(data_dir, wav_file),
            os.path.join(data_dir, wav_file.replace(".wav", ".txt")),
            images_dir_test,
            label_dir_test,
            i + 1,
        )

    images_dir_val = "data/images/val"
    label_dir_val = "data/labels/val"
    if not os.path.exists(images_dir_val):
        os.makedirs(images_dir_val)
    if not os.path.exists(label_dir_val):
        os.makedirs(label_dir_val)
    for i, wav_file in enumerate(tqdm(wav_files_val)):
        wav_to_image(
            os.path.join(data_dir, wav_file),
            os.path.join(data_dir, wav_file.replace(".wav", ".txt")),
            images_dir_val,
            label_dir_val,
            i + 1,
        )

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from PIL import Image
from typing import Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import librosa
import librosa.display


# def wav_to_image(
#     wav_filename: str,
#     label_filename: str,
#     images_dir: str,
#     label_dir: str,
#     output_fileid: int,
#     output_size: Tuple[int, int] = (640, 640),
#     cough_class: int = 0,
# ) -> Tuple[bool, Path, Path]:
#     """
#     Function to convert a WAV file to an image
#     Parameters
#     ----------
#     wav_filename : str
#         Path to the WAV file
#     label_filename : str
#         Path to the label file
#     images_dir : str
#         Path to the directory where the images will be saved
#     label_dir : str
#         Path to the directory where the labels will be saved
#     output_fileid : int
#         File ID to use for the output image
#     output_size : Tuple[int, int]
#         Size of the output image
#     cough_class : int
#         Class of the cough
#     Returns
#     -------
#     bool
#         True if the image was created successfully, False otherwise
#     """

#     # Read the WAV file
#     sample_rate, data = wavfile.read(wav_filename)

#     # Create the time axis for plotting (limited to 11 seconds)
#     if len(data) > 11 * sample_rate:
#         # If the audio is longer than 11 seconds, remove it
#         print(f"Audio file {wav_filename} is longer than 11 seconds. Skipping...")
#         return False, None, None
    
#     if len(data) < 512:  # same as n_fft
#         print(f"Audio file {wav_filename} is too short. Skipping...")
#         return False, None, None
    
#     audio_duration = len(data) / sample_rate
#     scale = 11 / audio_duration
#     # resizes the data to 11 seconds
#     time = np.linspace(0, 11, num=len(data))

#     data = data.astype(float)

#     mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40, n_fft=2048)
#     if mfccs.ndim == 3:
#         mfccs = mfccs[:, :, 0]
#     plt.figure(figsize=(10, 10))
#     librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
#     plt.axis("off")

#     # Save the plot to a file
#     plt.savefig(
#         Path(images_dir) / (str(output_fileid) + ".png"),
#         bbox_inches="tight",
#         pad_inches=0,
#         transparent=True,
#     )
#     plt.close()

#     # open fig and resize it to 640x640
#     img = Image.open(Path(images_dir) / (str(output_fileid) + ".png"))
#     # resize the image
#     img = img.resize(output_size)
#     # save the image
#     img.save(Path(images_dir) / (str(output_fileid) + ".png"))

#     # use scale to resize the interval
#     # first check if the txt file exists
#     if os.path.exists(label_filename):
#         with open(label_filename, "r") as f:
#             lines = f.readlines()
#             with open(Path(label_dir) / (str(output_fileid) + ".txt"), "w") as f2:
#                 for line in lines:
#                     start, end = line.split()
#                     start = int(float(start) / audio_duration * output_size[0])
#                     end = int(float(end) / audio_duration * output_size[0])
#                     assert start < end, f"start: {start}, end: {end}"
#                     # if we want to keep start/end
#                     # f2.write(
#                     #     str(start)
#                     #     + " "
#                     #     + str(end)
#                     #     + "\n"
#                     # )

#                     # if we want to keep the center of the interval
#                     f2.write(
#                         str(cough_class)
#                         + " "
#                         + str( float((start + end) // 2) / output_size[0])  # x center
#                         + " "
#                         + "0.5"  # y center
#                         + " "
#                         + str( float(end - start)  /  output_size[0])# width
#                         + " "
#                         + "1"  # height
#                         + "\n"
#                     )

#     else:
#         with open(Path(label_dir) / (str(output_fileid) + ".txt"), "w") as f2:
#             f2.write("")
#     return (
#         True,
#         Path(images_dir) / (str(output_fileid) + ".png"),
#         Path(label_dir) / (str(output_fileid) + ".txt"),
#     )
#     plt.close("all")

def wav_to_image(
    wav_filename: str,
    label_filename: str,
    images_dir: str,
    label_dir: str,
    output_fileid: int,
    output_size: Tuple[int, int] = (640, 640),
    cough_class: int = 0,
    max_duration: float = 11.0,
    n_fft: int = 2048,
    stereo_files: list = None,
) -> Tuple[bool, Path, Path]:

    sample_rate, data = wavfile.read(wav_filename)
    
    

    # Reject audio longer than max_duration
    if len(data) > max_duration * sample_rate:
        print(f"Audio file {wav_filename} is longer than {max_duration} seconds. Skipping...")
        return False, None, None

    if len(data) < n_fft:
        print(f"Audio file {wav_filename} is too short. Skipping...")
        return False, None, None
    
    # Convert stereo to mono if needed
    if data.ndim == 2:
        if stereo_files is not None:
            stereo_files.append(os.path.basename(wav_filename))
        data = data.mean(axis=1)

    # Pad with zeros if shorter than max_duration
    target_length = int(max_duration * sample_rate)
    if len(data) < target_length:
        padded_data = np.zeros(target_length, dtype=np.float32)
        padded_data[:len(data)] = data.astype(np.float32)
        data = padded_data
    else:
        data = data.astype(np.float32)

    audio_duration = len(data) / sample_rate 

    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40, n_fft=n_fft)
    if mfccs.ndim == 3:
        mfccs = mfccs[:, :, 0]

    # Save spectrogram as image
    plt.figure(figsize=(10, 10))
    librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
    plt.axis("off")
    plt.savefig(
        Path(images_dir) / (str(output_fileid) + ".png"),
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close()

    # Resize to 640x640
    img_path = Path(images_dir) / (str(output_fileid) + ".png")
    img = Image.open(img_path).resize(output_size)
    img.save(img_path)

    # Process label file
    if os.path.exists(label_filename):
        with open(label_filename, "r") as f:
            lines = f.readlines()
            with open(Path(label_dir) / (str(output_fileid) + ".txt"), "w") as f2:
                for line in lines:
                    start, end = line.split()
                    start = int(float(start) / max_duration * output_size[0])
                    end = int(float(end) / max_duration * output_size[0])
                    assert start < end, f"start: {start}, end: {end}"
                    f2.write(
                        f"{cough_class} "
                        f"{((start + end) // 2) / output_size[0]:.6f} "
                        f"0.5 "
                        f"{(end - start) / output_size[0]:.6f} "
                        f"1\n"
                    )
    else:
        with open(Path(label_dir) / (str(output_fileid) + ".txt"), "w") as f2:
            f2.write("")

    return True, img_path, Path(label_dir) / (str(output_fileid) + ".txt")


if __name__ == "__main__":
    # wav_to_image(
    #     "CoughSegmentation/Data/0a03da19-eb19-4f51-9860-78ad95fa8cb5.wav",
    #     "CoughSegmentation/Data/0a03da19-eb19-4f51-9860-78ad95fa8cb5.txt",
    #     "data/sample/images",
    #     "data/sample/labels",
    #     -1,
    # )

    # read in all the files
    data_dir = "data_w_noise_white0.03_nbn10_nbd0.5_w-beeping_w-talking"

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

    mapping = {}

    # create the directories
    images_dir = "data_mfcc_padding_11/images"
    label_dir = "data_mfcc_padding_11/labels"
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # loop through the files and create the images
    images_dir_train = "data_mfcc_padding_11/images/train"
    label_dir_train = "data_mfcc_padding_11/labels/train"
    if not os.path.exists(images_dir_train):
        os.makedirs(images_dir_train)
    if not os.path.exists(label_dir_train):
        os.makedirs(label_dir_train)
    stereo_audio_files = []
    for i, wav_file in enumerate(tqdm(wav_files_train)):
        success, image_path, label_path = wav_to_image(
            os.path.join(data_dir, wav_file),
            os.path.join(data_dir, wav_file.replace(".wav", ".txt")),
            images_dir_train,
            label_dir_train,
            i + 1,
            stereo_files=stereo_audio_files
        )
        if success:
            mapping[wav_file] = (str(image_path), str(label_path))

    images_dir_test = "data_mfcc_padding_11/images/test"
    label_dir_test = "data_mfcc_padding_11/labels/test"
    if not os.path.exists(images_dir_test):
        os.makedirs(images_dir_test)
    if not os.path.exists(label_dir_test):
        os.makedirs(label_dir_test)
    for i, wav_file in enumerate(tqdm(wav_files_test)):
        success, image_path, label_path = wav_to_image(
            os.path.join(data_dir, wav_file),
            os.path.join(data_dir, wav_file.replace(".wav", ".txt")),
            images_dir_test,
            label_dir_test,
            i + 1,
            stereo_files=stereo_audio_files
        )
        if success:
            mapping[wav_file] = (str(image_path), str(label_path))

    images_dir_val = "data_mfcc_padding_11/images/val"
    label_dir_val = "data_mfcc_padding_11/labels/val"
    if not os.path.exists(images_dir_val):
        os.makedirs(images_dir_val)
    if not os.path.exists(label_dir_val):
        os.makedirs(label_dir_val)
    for i, wav_file in enumerate(tqdm(wav_files_val)):
        success, image_path, label_path = wav_to_image(
            os.path.join(data_dir, wav_file),
            os.path.join(data_dir, wav_file.replace(".wav", ".txt")),
            images_dir_val,
            label_dir_val,
            i + 1,
            stereo_files=stereo_audio_files
        )
        if success:
            mapping[wav_file] = (str(image_path), str(label_path))
            
    print(stereo_audio_files)

    with open("data_mfcc_padding_11/mapping.json", "w") as f:
        json.dump(mapping, f)

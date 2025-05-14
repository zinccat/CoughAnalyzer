import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pydub import AudioSegment
from PIL import Image
from typing import Tuple
from pathlib import Path
import json


def mp3_to_wav(mp3_audio):
    """Convert an MP3 audio to WAV format in memory."""
    audio = AudioSegment.from_mp3(mp3_audio)
    samples = np.array(audio.get_array_of_samples())

    # Convert stereo to mono
    if audio.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)

    return samples.astype(np.float32), audio.frame_rate


def wav_to_mfcc_image(
    mp3_filename: str,
    images_dir: str,
    output_filename: str,
    output_size: Tuple[int, int] = (640, 640),
    n_mfcc: int = 40,
    n_fft: int = 2048,
    max_duration: float = 11.0,  # seconds
) -> Tuple[bool, Path]:

    data, sample_rate = mp3_to_wav(mp3_filename)
    max_len = int(max_duration * sample_rate)

    # Truncate or pad to 11 seconds
    if len(data) > max_len:
        data = data[:max_len]
    elif len(data) < max_len:
        padded = np.zeros(max_len, dtype=np.float32)
        padded[:len(data)] = data
        data = padded

    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft)

    if mfccs.ndim == 3:
        mfccs = mfccs[:, :, 0]

    # Plot MFCCs
    plt.figure(figsize=(10, 10))
    librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
    plt.axis("off")

    output_path = Path(images_dir) / (output_filename + ".png")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close()

    # Resize image canvas (not the MFCC resolution)
    img = Image.open(output_path).resize(output_size)
    img.save(output_path)

    return True, output_path


if __name__ == "__main__":
    images_dir = "/n/netscratch/kung_lab/Everyone/llm_hdc/cough/cough_mp3_mfcc_11s"
    for i in range(4,13):
        data_dir = f"/n/netscratch/kung_lab/Everyone/llm_hdc/cough/cough_mp3/Files_{i}_mp3"
        mp3_files = [f for f in os.listdir(data_dir) if f.endswith(".mp3")]
        
        os.makedirs(images_dir, exist_ok=True)
        mapping = {}

        for mp3_file in mp3_files:
            try:
                output_filename = os.path.splitext(mp3_file)[0]
                success, image_path = wav_to_mfcc_image(
                    os.path.join(data_dir, mp3_file),
                    images_dir,
                    output_filename,
                )
                if success:
                    mapping[mp3_file] = str(image_path)
            except Exception as e:
                print(f"Skipping {mp3_file}: {e}")
            
                

    # with open("data_inference_mfcc_padding_11/mapping_inference.json", "w") as f:
    #     json.dump(mapping, f)

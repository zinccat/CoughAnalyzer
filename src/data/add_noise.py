import os
import numpy as np
import random
from pydub import AudioSegment
import argparse
import shutil


def add_noise(
    pink,
    brown,
    white_level,
    pink_level,
    brown_level,
    noise_burst_num,
    noise_burst_duration,
    beeping,
    talking,
):
    # Set the path to your folder
    folder_path = "./Data/"

    # Get all files in the folder
    files = os.listdir(folder_path)

    # Separate files by extension
    wav_files = {os.path.splitext(f)[0] for f in files if f.endswith(".wav")}
    txt_files = {os.path.splitext(f)[0] for f in files if f.endswith(".txt")}
    # Find matching base names
    matching_files = wav_files
    # Store the corresponding filenames
    coughing_files = [f"{name}.wav" for name in matching_files]

    if talking:
        unmatching_files = wav_files - txt_files
        talking_files = [f"{name}.wav" for name in unmatching_files]

    output_dir = (
        "data_w_noise"
        + (f"_white{white_level}")
        + (f"_pink{pink_level}" if pink else "")
        + (f"_brown{brown_level}" if brown else "")
        + (f"_nbn{noise_burst_num}")
        + (f"_nbd{noise_burst_duration}")
        + ("_w-beeping" if beeping else "_wo-beeping")
        + ("_w-talking" if talking else "_wo-talking")
    )

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for base_name in matching_files:
        src_txt = os.path.join(folder_path, base_name + ".txt")
        dst_txt = os.path.join(output_dir, base_name + ".txt")
        if os.path.exists(src_txt):
            shutil.copy(src_txt, dst_txt)

    if beeping:
        beep_file = "beeping.wav"
        beep_sound = AudioSegment.from_wav(beep_file)
    n = 0
    for input_file in coughing_files:
        n += 1
        # Load base audio using pydub
        output_file = os.path.join(output_dir, f"{input_file}")
        input_file_path = os.path.join(folder_path, input_file)
        base_audio = AudioSegment.from_file(input_file_path, format="wav")

        # Function to generate different types of noise as an AudioSegment
        def generate_noise(duration_ms, sample_rate, channels, noise_type="white"):
            num_samples = int(sample_rate * (duration_ms / 1000))

            if noise_type == "white":
                noise = np.random.normal(0, white_level, num_samples)
            elif noise_type == "pink":
                white = np.random.normal(0, pink_level, num_samples)
                pink = np.cumsum(white)
                pink = pink / np.max(np.abs(pink))  # Normalize
                noise = pink
            elif noise_type == "brown":
                brown = np.cumsum(np.random.normal(0, brown_level, num_samples))
                brown = brown / np.max(np.abs(brown))  # Normalize
                noise = brown
            else:
                noise = np.zeros(num_samples)

            # Convert noise to int16 format and create an AudioSegment
            noise_int16 = (noise * 32767).astype(np.int16)
            return AudioSegment(
                noise_int16.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,  # int16 = 2 bytes
                channels=channels,
            )

        # Add random noise bursts at different positions
        burst_duration_ms = int(
            noise_burst_duration * 1000
        )  # Convert duration to milliseconds

        for _ in range(noise_burst_num):
            start_ms = random.randint(0, len(base_audio) - burst_duration_ms)
            noise_type = random.choice(
                ["white"] + (["pink"] if pink else []) + (["brown"] if brown else [])
            )  # white is always there
            noise_segment = generate_noise(
                burst_duration_ms,
                base_audio.frame_rate,
                base_audio.channels,
                noise_type,
            )
            base_audio = base_audio.overlay(noise_segment, position=start_ms)

        if beeping:
            # Overlay beep sound at the beginning
            beep_duration = min(
                len(beep_sound), len(base_audio)
            )  # Limit beep to base audio length
            beep_trimmed = beep_sound[:beep_duration]  # Trim beep if needed
            base_audio = base_audio.overlay(beep_trimmed, position=0)

        if talking:
            # Overlay a randomly selected talking audio
            talking_file = os.path.join(folder_path, random.choice(talking_files))
            print(talking_file)
            talking_sound = AudioSegment.from_file(talking_file, format="wav")
            talking_duration = min(len(talking_sound), len(base_audio))
            talking_trimmed = talking_sound[:talking_duration]
            base_audio = base_audio.overlay(talking_trimmed, position=0)

        # Export final processed audio
        base_audio.export(output_file, format="wav")
        print(
            f"Processed audio with noise (white, pink, brown) and beeps saved to '{output_file}'"
        )
    print(n)


# Create argument parser
parser = argparse.ArgumentParser(description="Process user input from CLI.")

# Add arguments
parser.add_argument("--pink", type=str, help="Do you want pink noise", default=False)
parser.add_argument("--brown", type=str, help="Do you want brown noise", default=False)
parser.add_argument(
    "--white_level", type=str, help="Level of white noise", default=0.01
)
parser.add_argument("--pink_level", type=str, help="Level of pink noise", default=0.01)
parser.add_argument(
    "--brown_level", type=str, help="Level of brown noise", default=0.01
)
parser.add_argument(
    "--noise_burst_num", type=str, help="Number of noise bursts", default=20
)
parser.add_argument(
    "--noise_burst_duration", type=str, help="Duration of each noise burst", default=1.0
)
parser.add_argument(
    "--beeping", type=str, help="Do you want beeping noise", default=False
)
parser.add_argument(
    "--talking", type=str, help="Do you want talking noise", default=False
)

args = parser.parse_args()
add_noise(
    pink=(args.pink == "True"),
    brown=(args.brown == "True"),
    white_level=float(args.white_level),
    pink_level=float(args.pink_level),
    brown_level=float(args.brown_level),
    noise_burst_num=int(args.noise_burst_num),
    noise_burst_duration=float(args.noise_burst_duration),
    beeping=(args.beeping == "True"),
    talking=(args.talking == "True"),
)

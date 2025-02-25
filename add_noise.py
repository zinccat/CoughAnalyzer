import os
import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf

# Set the path to your folder
folder_path = "./CoughSegmentation/Data/"

# Get all files in the folder
files = os.listdir(folder_path)

# Separate files by extension
wav_files = {os.path.splitext(f)[0] for f in files if f.endswith(".wav")}
txt_files = {os.path.splitext(f)[0] for f in files if f.endswith(".txt")}

# Find matching base names
matching_files = wav_files & txt_files

# Store the corresponding filenames
audio_files = [f"{name}.wav" for name in matching_files]
# label_files = [f"./CoughSegmentation/Data/{name}.txt" for name in matching_files]


output_dir = "data_w_noise"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for input_file in audio_files:
  # Read the WAV file
  output_file = os.path.join(output_dir, f"{input_file}_w_noise.wav")
  input_file = f"./CoughSegmentation/Data/{input_file}"
  sample_rate, audio_data = wav.read(input_file)

  # Ensure audio data is in float format for proper processing
  if audio_data.dtype != np.float32:
      audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

  # Function to generate different types of noise
  def generate_noise(length, noise_type="white"):
      if noise_type == "white":
          return np.random.normal(0, 0.02, length)  # White noise
      elif noise_type == "pink":
          # Approximate pink noise by filtering white noise
          white = np.random.normal(0, 0.02, length)
          pink = np.cumsum(white)  # Integration makes it pink-like
          return pink / np.max(np.abs(pink))
      elif noise_type == "brown":
          # Brown noise is even lower-frequency than pink
          brown = np.cumsum(np.random.normal(0, 0.01, length))
          return brown / np.max(np.abs(brown))
      else:
          return np.zeros(length)

  # Randomly scatter noise throughout the audio
  num_bursts = 20  # Number of noise bursts
  burst_duration = int(1 * sample_rate)  # 1 second per burst

  for _ in range(num_bursts):
      start = np.random.randint(0, len(audio_data) - burst_duration)
      noise_type = np.random.choice(["white", "pink", "brown"])
      noise_burst = generate_noise(burst_duration, noise_type)
      audio_data[start:start + burst_duration] += noise_burst

  # Ensure the values stay within the valid range [-1, 1]
  audio_data = np.clip(audio_data, -1.0, 1.0)

  # Save the output as a new WAV file
  sf.write(output_file, audio_data, sample_rate)

import tensorflow_hub as hub
import numpy as np
import librosa
import tensorflow as tf
from collections import defaultdict
import argparse

def cough_detection(audio_filename):
    
    # Load audio
    audio_data, sr = librosa.load(audio_filename, sr=16000)  # YAMNet needs 16kHz

    # Load the YAMNet model
    yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
    yamnet_model = hub.load(yamnet_model_handle)

    # Run prediction
    scores, embeddings, spectrogram = yamnet_model(audio_data)

    # Get class names
    import requests
    class_map_path = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
    class_map = requests.get(class_map_path).text.splitlines()

    import pandas as pd
    import io  # <-- Add this import

    df = pd.read_csv(io.StringIO("\n".join(class_map)))

    # YAMNet output time per frame (0.96 seconds per frame)

    frame_duration = 0.96  # duration of each frame in seconds
    # start_time_stamps = (np.arange(0, len(audio_data) / sr, frame_duration / 2)[:-1])  # Start time stamps for frames

    target_class = "Cough"

    detection_times_in_secs = []
    num_frames = scores.shape[0]
    for i in range(num_frames):
      # check if target class is within the top three most likely class
      timestamp = i * (frame_duration / 2)
      top_three_class_indices = np.argsort(scores[i, :])[-3:]
      index_of_target_class = df[df['display_name'] == target_class]['index'].values[0]

      if index_of_target_class in top_three_class_indices:
        detection_times_in_secs.append(float(timestamp))


    # Print out the times for each detected class
    print(f"{target_class} detected at these times (in seconds):", detection_times_in_secs)
    return detection_times_in_secs



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a specific server instance.")
    parser.add_argument("--audio_filename", type=str, required=True, help="audio filename")
    args = parser.parse_args()
    cough_detection(args.audio_filename)
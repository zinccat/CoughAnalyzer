from cough_detection import cough_detection
import librosa
import numpy as np
import argparse
import os
import pandas as pd
sr=16000

def extract_cough_segments(audio_filename, detection_times):
    audio_data, sr = librosa.load(audio_filename, sr=16000)  # YAMNet needs 16kHz
    # Segment length
    frame_duration = 0.96

    # Generate intervals (start, end) for each segment
    intervals = [(start, start + frame_duration) for start in detection_times]

    # Sort intervals by start time
    intervals.sort()

    # Merge overlapping intervals
    merged_intervals = []
    for interval in intervals:
        if not merged_intervals or merged_intervals[-1][1] < interval[0]:
            merged_intervals.append(interval)
        else:
            merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], interval[1]))

    cough_segments = []
    for start_time, end_time in merged_intervals:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        cough_segment = audio_data[start_sample:end_sample]
        cough_segments = np.concatenate([cough_segments, cough_segment]) 
    
    return cough_segments

def extract_features(audio_filename, audio_segment):
    # Extract features like MFCCs from each segment
    mfcc = librosa.feature.mfcc(y=np.array(audio_segment), sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc, axis=1)  # Take the mean of MFCC features across time frames
    return mfcc


def save_cough_features(audio_directory, output_filename):
    df = pd.DataFrame(columns=[f'feature {i}' for i in range(1, 14)])

    for root, dirs, files in os.walk(audio_directory):
        for file in files:
            if file.endswith('.mp3'):
                audio_filename = os.path.join(root, file)
                print(audio_filename)
                detection_times = cough_detection(audio_filename)
                if len(detection_times) > 0:
                    cough_segments = extract_cough_segments(audio_filename, detection_times)
                    feature_row = extract_features(audio_filename, cough_segments)
                    df.loc[audio_filename] = feature_row 
    df.to_csv(f'{output_filename}.csv', index=True, header=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a specific server instance.")
    parser.add_argument("--audio_directory", type=str, required=True, help="audio files directory")
    parser.add_argument("--output_filename", type=str, required=True, help="csv output filename")
    args = parser.parse_args()
    save_cough_features(args.audio_directory, args.output_filename)
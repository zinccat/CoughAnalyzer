from cough_detection import cough_detection
import librosa
import numpy as np
import argparse
import os
import pandas as pd
sr=16000

def extract_cough_segment_intervals(audio_filename, detection_times):
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

    return merged_intervals

def save_cough_intervals(audio_directory, output_filename, top_n):
    df = pd.DataFrame(columns=['intervals'])

    for root, dirs, files in os.walk(audio_directory):
        for file in files:
            if file.endswith('.wav'): #change to '.wav' to analyze the CoughSegmentation dataset
                audio_filename = os.path.join(root, file)
                print(audio_filename)
                detection_times = cough_detection(audio_filename, top_n)
                if len(detection_times) > 0:
                    cough_intervals = extract_cough_segment_intervals(audio_filename, detection_times)
                    df.loc[audio_filename] = [cough_intervals] 
    df.to_csv(f'{output_filename}.csv', index=True, header=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a specific server instance.")
    parser.add_argument("--audio_directory", type=str, required=True, help="audio files directory")
    parser.add_argument("--output_filename", type=str, required=True, help="csv output filename")
    parser.add_argument("--top_n", type=str, required=True, help="csv output filename")
    args = parser.parse_args()
    save_cough_intervals(args.audio_directory, args.output_filename, int(args.top_n))
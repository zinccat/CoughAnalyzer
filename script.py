import os
import re
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# Define the root directory
root_dir = "cough_mp3_full"

# Dictionary to store patient counts and folder counts
patient_counts = defaultdict(int)
folder_counts = defaultdict(int)
recording_lengths = []

# Traverse the directory structure
for folder in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, folder)
    
    if os.path.isdir(folder_path):  # Ensure it's a folder
        folder_counts[folder] = len(os.listdir(folder_path))  # Count files in folder
        
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            match = re.match(r"(X\d+)", file)  # Extract patient ID from filename
            if match:
                patient_id = match.group(1)
                patient_counts[patient_id] += 1  # Count recordings per patient

            # Extract audio length
            try:
                audio, sr = librosa.load(file_path, sr=None)
                duration = librosa.get_duration(y=audio, sr=sr)
                recording_lengths.append(duration)
            except Exception as e:
                print(f"Could not process {file}: {e}")

# Convert to DataFrame for easy visualization
df_patient_counts = pd.DataFrame(list(patient_counts.items()), columns=["Patient_ID", "Recording_Count"])
df_folder_counts = pd.DataFrame(list(folder_counts.items()), columns=["Folder", "File_Count"])

# Save results to CSV
df_patient_counts.to_csv("patient_recording_counts.csv", index=False)
df_folder_counts.to_csv("folder_recording_counts.csv", index=False)

# Save patient recording counts to a text file instead of printing everything
with open("patient_recording_counts.txt", "w") as f:
    f.write(df_patient_counts.to_string(index=False))

# Display only summary in terminal
total_patients = len(df_patient_counts)
min_recordings = df_patient_counts["Recording_Count"].min()
max_recordings = df_patient_counts["Recording_Count"].max()
most_common_count = df_patient_counts["Recording_Count"].mode()[0]

print("\n===== Dataset Summary =====")
print(f"Total unique patients: {total_patients}")
print(f"Minimum recordings per patient: {min_recordings}")
print(f"Maximum recordings per patient: {max_recordings}")
print(f"Most common recording count: {most_common_count}")

# Identify missing or duplicate recordings
missing_patients = df_patient_counts[df_patient_counts["Recording_Count"] < most_common_count]
extra_patients = df_patient_counts[df_patient_counts["Recording_Count"] > most_common_count]

if not missing_patients.empty:
    print(f"Patients with missing recordings: {len(missing_patients)} (see patient_recording_counts.txt)")
if not extra_patients.empty:
    print(f"Patients with extra recordings: {len(extra_patients)} (see patient_recording_counts.txt)")

# Generate histogram for recording lengths
plt.figure(figsize=(10, 5))
plt.hist(recording_lengths, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("Recording Length (seconds)")
plt.ylabel("Number of Recordings")
plt.title("Histogram of Recording Lengths")
plt.grid(axis="y")
plt.show()

# Generate histogram for number of recordings per patient
plt.figure(figsize=(10, 5))
plt.hist(df_patient_counts["Recording_Count"], bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("Number of Recordings per Patient")
plt.ylabel("Number of Patients")
plt.title("Histogram of Recordings per Patient")
plt.grid(axis="y")
plt.show()

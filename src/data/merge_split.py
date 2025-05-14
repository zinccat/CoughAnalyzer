import os
import argparse
from pydub import AudioSegment

def process_audio_files(audio_folder, text_folder, output_folder):
    """
    Process audio files by cutting segments according to timestamps in text files.
    
    Args:
        audio_folder (str): Path to folder containing MP3 files
        text_folder (str): Path to folder containing text files with timestamps
        output_folder (str): Path to save processed audio files
    """
    cnt = 0
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get list of mp3 files
    mp3_files = [f for f in os.listdir(audio_folder) if f.endswith('.mp3')]
    
    for mp3_file in mp3_files:
        base_name = os.path.splitext(mp3_file)[0]
        txt_file = os.path.join(text_folder, base_name + '.txt')
        
        # Check if corresponding text file exists
        if not os.path.exists(txt_file):
            print(f"No text file found for {mp3_file}, skipping...")
            continue
        
        # Load the audio file
        print(f"Processing {mp3_file}...")
        audio_path = os.path.join(audio_folder, mp3_file)
        audio = AudioSegment.from_mp3(audio_path)
        
        # Read timestamps from text file
        segments = []
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        start_time, end_time = map(float, line.split())
                        # Convert seconds to milliseconds for pydub
                        start_ms = int(start_time * 1000)
                        end_ms = int(end_time * 1000)
                        segment = audio[start_ms:end_ms]
                        segments.append(segment)
                    except ValueError:
                        print(f"Invalid line in {txt_file}: {line}, skipping...")
        
        if segments:
            # Concatenate all segments
            final_audio = segments[0]
            for segment in segments[1:]:
                final_audio += segment
            
            # Export the final audio
            output_path = os.path.join(output_folder, f"{base_name}.mp3")
            final_audio.export(output_path, format="mp3")
            print(f"Created {output_path}")
            cnt += 1
        else:
            print(f"No valid segments found in {txt_file}")
        
    print(f"Processed {cnt} files so far...")
    print(len(mp3_files))

def main():
    parser = argparse.ArgumentParser(description='Process audio files based on timestamp segments.')
    parser.add_argument('--audio', required=True, help='Folder containing MP3 files')
    parser.add_argument('--text', required=True, help='Folder containing text files with timestamps')
    parser.add_argument('--output', required=True, help='Folder to save processed audio files')
    
    args = parser.parse_args()
    
    process_audio_files(args.audio, args.text, args.output)

if __name__ == "__main__":
    main()
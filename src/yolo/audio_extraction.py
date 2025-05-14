import os
from ultralytics import YOLO
from pathlib import Path
from pydub import AudioSegment
from PIL import Image
import json
from tqdm import tqdm

def get_bounding_boxes(results):
    boxes = []
    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():
            x_min, y_min, x_max, y_max = box[:4]
            boxes.append((int(x_min), int(x_max)))
    return boxes

def extract_cough_clip_from_mp3(mp3_path, output_path, x_min, x_max, image_width=640, max_duration=11.0):
    audio = AudioSegment.from_mp3(mp3_path)
    start_ms = (x_min / image_width) * max_duration * 1000
    end_ms = (x_max / image_width) * max_duration * 1000
    cough_clip = audio[start_ms:end_ms]
    cough_clip.export(output_path, format="mp3")

def run_yolo_and_extract_audio(model_path, images_dir, mp3_root_dir, output_audio_dir, limit=None, batch_size=100):
    model = YOLO(model_path)
    Path(output_audio_dir).mkdir(parents=True, exist_ok=True)

    image_paths = sorted([
        os.path.join(images_dir, img)
        for img in os.listdir(images_dir)
        if img.lower().endswith(".png")
    ])
    if limit:
        image_paths = image_paths[:limit]

    metadata = {}

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i + batch_size]
        results = model(batch_paths)

        for j, result in enumerate(results):
            image_path = batch_paths[j]
            image_filename = os.path.basename(image_path)
            file_id = image_filename.replace(".png", "")

            # Match MP3
            for subfolder in os.listdir(mp3_root_dir):
                candidate_path = os.path.join(mp3_root_dir, subfolder, file_id + ".mp3")
                if os.path.exists(candidate_path):
                    mp3_path = candidate_path
                    break
            else:
                continue  # no match

            boxes = get_bounding_boxes([result])
            if not boxes:
                continue

            for idx, (x_min, x_max) in enumerate(boxes):
                output_mp3 = os.path.join(output_audio_dir, f"{file_id}_cough{idx + 1}.mp3")
                extract_cough_clip_from_mp3(mp3_path, output_mp3, x_min, x_max)
                metadata.setdefault(file_id, []).append({
                    "bbox": [x_min, x_max],
                    "output": output_mp3
                })

    with open(os.path.join(output_audio_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Finished processing all batches.")


# Example usage
if __name__ == "__main__":
    model_path = '/n/home04/huandongchang/CoughAnalyzer/src/yolo/runs/detect/train3/weights/best.pt'
    images_dir = '/n/netscratch/kung_lab/Everyone/llm_hdc/cough/cough_mp3_mfcc_11s'
    mp3_root_dir = '/n/netscratch/kung_lab/Everyone/llm_hdc/cough/cough_mp3'
    output_audio_dir = '/n/netscratch/kung_lab/Everyone/llm_hdc/cough/yolo_cough_audio_all'

    run_yolo_and_extract_audio(model_path, images_dir, mp3_root_dir, output_audio_dir, limit=None)

import gradio as gr
import os
import numpy as np
# restore the old alias so librosa will import
np.complex = complex
import librosa
import soundfile as sf
import tempfile
import torch
from ultralytics import YOLO
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from pydub import AudioSegment
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Paths to your models/weights
YOLO_WEIGHTS_PATH = "./src/yolo/runs/detect/train2/weights/best."
W2V_MODEL_DIR     = "./checkpoint-1696"

# Load models globally for efficiency
yolo_model        = YOLO(YOLO_WEIGHTS_PATH, verbose=False)
feature_extractor = AutoFeatureExtractor.from_pretrained(W2V_MODEL_DIR, local_files_only=True)
classifier        = AutoModelForAudioClassification.from_pretrained(W2V_MODEL_DIR, local_files_only=True)

def merge_intervals(intervals):
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for start, end in intervals:
        if not merged or merged[-1][1] < start:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged

def process(audio_path):
    # Convert to WAV if needed
    tmp_dir = tempfile.gettempdir()
    ext = os.path.splitext(audio_path)[1].lower()
    if ext != ".wav":
    # load & re-save as WAV without FFmpeg
        y_tmp, sr_tmp = librosa.load(audio_path, sr=16000)
        converted = os.path.join(tmp_dir, "input_converted.wav")
        sf.write(converted, y_tmp, sr_tmp)
        audio_path = converted


    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)
    duration = len(y) / sr

    # === Strictly follow his WAV-to-image method ===
    # Generate time array
    time = np.linspace(0, duration, num=len(y))

    # Plot waveform
    plt.figure(figsize=(10, 10))
    plt.plot(time, y, color="black")
    plt.xlim(0, duration)
    plt.ylim(np.min(y), np.max(y))
    plt.grid(False)
    plt.axis("off")

    # Save the figure
    tmp_png = os.path.join(tmp_dir, "waveform_tmp.png")
    plt.savefig(tmp_png, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close()

    # Open, convert to gray, resize
    final_png = os.path.join(tmp_dir, "waveform.png")
    img = Image.open(tmp_png).convert("L")
    img = img.resize((640, 640))
    img.save(final_png)

    # Run YOLO segmentation
    results = yolo_model(final_png)
    seg_img = results[0].plot()[..., ::-1]

    # Map boxes to time intervals
    boxes = results[0].boxes.xyxy.cpu().numpy()
    img_h, img_w, _ = seg_img.shape
    intervals = [(x1/img_w*duration, x2/img_w*duration) for x1, _, x2, _ in boxes]
    merged_list = merge_intervals(intervals)

    # Merge segments or use full audio
    if merged_list:
        merged_audio = np.concatenate([y[int(s*sr):int(e*sr)] for s, e in merged_list])
    else:
        merged_audio = y

    # Save merged/full audio
    merged_path = os.path.join(tmp_dir, "merged_audio.wav")
    sf.write(merged_path, merged_audio, sr)
    if len(merged_audio) < 10:
        merged_audio = np.pad(merged_audio, (0, 10 - len(merged_audio)))

    # Classification
    inputs = feature_extractor(merged_audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = classifier(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    label_map = {0: "non-cough", 1: "cough"}
    result_label = label_map.get(pred, str(pred))

    # Return the waveform image path for display, plus other outputs
    return final_png, audio_path, seg_img, merged_path, result_label

# Build Gradio interface
def app():
    with gr.Blocks() as demo:
        gr.Markdown("## Cough Detection and Analysis Pipeline")
        with gr.Row():
            with gr.Column(scale=0.5):
                input_audio     = gr.Audio(type="filepath", label="Input Audio")
            with gr.Column(scale=0.5):
                waveform_img = gr.Image(label="Waveform (640×640)")
                original_player = gr.Audio(label="Original Audio", type="filepath")
            with gr.Column(scale=0.5):
                yolo_image    = gr.Image(label="YOLO Segmentation", type="numpy")
                merged_player = gr.Audio(label="Processed Audio (Cough)", type="filepath")
            with gr.Column(scale=0.5):
                classification= gr.Textbox(label="Classification Result")
        process_btn = gr.Button("Analyze")
        process_btn.click(
            fn=process,
            inputs=[input_audio],
            outputs=[waveform_img, original_player, yolo_image, merged_player, classification]
        )
    return demo

if __name__ == "__main__":
    app().launch()

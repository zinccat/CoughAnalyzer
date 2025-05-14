import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO('./src/yolo/runs/detect/train2/weights/best.pt')

def yolo_segment(image):
    img_array = np.uint8(image)
    img = Image.fromarray(img_array)
    results = model.predict(img)
    annotated_img = results[0].plot()
    annotated_img = annotated_img[:, :, ::-1]

    return annotated_img 

interface = gr.Interface(
    fn=yolo_segment,
    inputs=gr.Image(),
    outputs=gr.Image(),
    title="YOLO segmentation",
    description="upload your picture to get YOLO segmentation result"
)

interface.launch()

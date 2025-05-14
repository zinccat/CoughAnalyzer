import os
from ultralytics import YOLO
from pathlib import Path

def run_yolo_inference(model_path, images_dir, output_dir, limit=None):
    # Load the YOLO model
    model = YOLO(model_path)

    # Create output directory if it does not exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Gather list of image paths
    image_paths = [os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))] 

    if limit != None:
        image_paths = image_paths[:limit]
        
    # Run batched inference on a list of images
    results = model(image_paths)  # Returns a list of Results objects

    # Process results list
    for i, result in enumerate(results):
        filename = os.path.basename(image_paths[i])
        save_path = os.path.join(output_dir, filename)
        
        # Show result in a window (optional, comment out if running on a headless server)
        # result.show()

        # Save result to disk
        result.save(filename=save_path)

        print(f"Processed and saved results to {save_path}")

if __name__ == "__main__":
    model_path = './runs/detect/train3/weights/best.pt'
    images_dir = '../../data_inference_mfcc_padding_11/images/File1'
    output_dir = '../../data_inference_mfcc_padding_11/yolo_inference_train3/File1'
    
    run_yolo_inference(model_path, images_dir, output_dir, 30)

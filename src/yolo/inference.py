# import os
# import numpy as np
# from PIL import Image, ImageDraw
# from pathlib import Path

# # Import YOLO from the Ultralytics package
# from ultralytics import YOLO

# def run_yolo_inference(model_path, images_dir, output_dir):
#     # Load the YOLO model
#     model = YOLO(model_path)

#     # Create output directory if it does not exist
#     Path(output_dir).mkdir(parents=True, exist_ok=True)

#     # Process each image in the directory
#     for img_name in os.listdir(images_dir):
#         img_path = os.path.join(images_dir, img_name)
#         if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
#             # Load image
#             img = Image.open(img_path)

#             # Perform inference
#             results = model.predict(img)  # Adjust size as per your requirements
            
#             # Check the type and structure of results
#             print(type(results))  # See what type is returned
#             print(results)        # See the content of the results

#             # Visualize and save the image with detections
#             save_path = os.path.join(output_dir, img_name)
#             if isinstance(results, list):  # Assuming results are list of detections
#                 draw = ImageDraw.Draw(img)
#                 for detection in results:
#                     # Example assuming detection is a tuple or list in the form (xmin, ymin, xmax, ymax)
#                     draw.rectangle(detection[:4], outline="red", width=2)
#                 img.save(save_path)
#             else:
#                 # If results have a save function or similar, use it
#                 results.save(save_path)  # This saves the image with boxes

#             print(f"Processed and saved {img_name}")

# if __name__ == "__main__":
#     model_path = './runs/detect/train/weights/best.pt'  # Update to your model's specific path
#     images_dir = '../../data_inference/images/File1'
#     output_dir = '../../data_inference/yolo_inference/File1'
    
#     run_yolo_inference(model_path, images_dir, output_dir)



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
    model_path = './runs/detect/train/weights/best.pt'
    images_dir = '../../data_inference/images/File1'
    output_dir = '../../data_inference/yolo_inference/File1'
    
    run_yolo_inference(model_path, images_dir, output_dir, 20)

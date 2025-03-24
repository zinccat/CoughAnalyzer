import os
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

def merge_intervals(intervals):
    """
    Merge overlapping intervals.
    
    Parameters:
        intervals (list of tuple): List of intervals as (x_min, x_max).
    
    Returns:
        List of merged intervals as [[start, end], ...]
    """
    # Sort intervals based on the start coordinate
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            # No overlap; add new interval
            merged.append(list(interval))
        else:
            # Overlap exists; merge with the last interval
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged

def crop_and_concat(image_path, bboxes):
    """
    Crops regions from an image using merged x-ranges from bounding boxes and concatenates them horizontally.
    The full height of the image is preserved.
    
    Parameters:
        image_path (str): Path to the input image.
        bboxes (list of tuple): List of bounding boxes in (x1, y1, x2, y2) format.
    
    Returns:
        Image: A new PIL Image that is the horizontal concatenation of the cropped regions.
    """
    # Open the image
    image = Image.open(image_path)
    width, height = image.size

    # Extract the x intervals from bounding boxes
    x_intervals = [(bbox[0], bbox[2]) for bbox in bboxes]
    if not x_intervals:
        # No bounding boxes found, return the original image
        return image

    # Merge overlapping x intervals
    merged_intervals = merge_intervals(x_intervals)
    
    # Crop regions using the merged x intervals and the full image height
    crops = []
    for x_start, x_end in merged_intervals:
        crop = image.crop((int(x_start), 0, int(x_end), height))
        crops.append(crop)
    
    # Calculate total width for the new image and create a blank image
    total_width = sum(crop.width for crop in crops)
    new_image = Image.new('RGB', (total_width, height))
    
    # Paste the cropped regions side-by-side
    current_x = 0
    for crop in crops:
        new_image.paste(crop, (current_x, 0))
        current_x += crop.width
    
    return new_image

def process_folder(input_folder, output_folder, model_path):
    """
    Process all images in a folder with YOLO detection, crop the image based on merged x-intervals,
    and save the concatenated result.
    
    Parameters:
        input_folder (str): Folder containing input images.
        output_folder (str): Folder to save processed images.
        model_path (str): Path to the YOLO model weights.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize YOLO model
    model = YOLO(model_path, verbose=False)
    
    # Supported image extensions
    supported_ext = [".png", ".jpg", ".jpeg", ".bmp"]
    
    # List all image files in the input folder
    image_files = [
        os.path.join(input_folder, f) for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in supported_ext
    ]
    
    if not image_files:
        print("No image files found in the input folder.")
        return
    
    # Process each image individually
    for image_path in tqdm(image_files):
        # print(f"Processing {image_path} ...")
        # Run detection on the image
        results = model(image_path)
        # YOLO returns a list of results; each result has boxes.xyxy (x1, y1, x2, y2)
        # Convert to a list of tuples for processing
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes.xyxy is not None else []
        bboxes = [tuple(box) for box in boxes]
        
        # Crop and concatenate based on merged x intervals
        result_image = crop_and_concat(image_path, bboxes)
        
        # Save the processed image to the output folder
        base_name = os.path.basename(image_path)
        out_path = os.path.join(output_folder, f"{base_name}")
        result_image.save(out_path)
        # print(f"Saved processed image to {out_path}")

if __name__ == "__main__":
    # Set the paths for the YOLO model, input folder, and output folder
    model_path = "src/yolo/runs/detect/train/weights/best.pt"
    input_folder = "data/coughvid_images/train"
    output_folder = "data/coughvid_images_cropped/train"
    
    process_folder(input_folder, output_folder, model_path)

    input_folder = "/home/zinccat/codes/CoughAnalyzer/data/coughvid_images/val"
    output_folder = "/home/zinccat/codes/CoughAnalyzer/data/coughvid_images_cropped/val"
    process_folder(input_folder, output_folder, model_path)

    input_folder = "/home/zinccat/codes/CoughAnalyzer/data/coughvid_images/test"
    output_folder = "/home/zinccat/codes/CoughAnalyzer/data/coughvid_images_cropped/test"
    process_folder(input_folder, output_folder, model_path)
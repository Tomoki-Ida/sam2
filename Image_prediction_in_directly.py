# Set-up
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Ensure environment variables are set before importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    print("CUDA is not available, using CPU.")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "/nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
    
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# Load bounding boxes and keypoints from JSON file
def load_annotations(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Create a mapping from image_id to file_name
    id_to_filename = {image['id']: image['file_name'] for image in data['images']}

    annotations = {}
    for entry in data['annotations']:
        image_id = entry['image_id']
        file_name = id_to_filename.get(image_id)
        if file_name:
            bbox = entry['bbox']  # Assuming 'bbox' contains [x1, y1, x2, y2]
            keypoints = entry.get('keypoints', [])  # Assuming 'keypoints' is in the JSON
            annotations[file_name] = {'bbox': bbox, 'keypoints': keypoints}
        else:
            print(f"No file name found for image_id: {image_id}")
    return annotations

# Process keypoints to extract coordinates and visibility
def process_keypoints(keypoints):
    """
    Process nested keypoints to extract visible coordinates and their labels.

    :param keypoints: List of lists, where each sublist is [x, y, visibility].
    :return: A list of coordinates and labels for visible keypoints.
    """
    coords = []
    labels = []
    try:
        for keypoint in keypoints:
            if isinstance(keypoint, list) and len(keypoint) == 3:
                x, y, v = keypoint
                if v > 0:  # Only use visible keypoints
                    coords.append((x, y))
                    labels.append(1)  # Visible point label
                else:
                    print(f"Skipping keypoint with visibility: {v}")
            else:
                print(f"Invalid keypoint format: {keypoint}")
    except Exception as e:
        print(f"Error processing keypoints: {keypoints}, error: {e}")
    return coords


# Overlay functions
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.random.random(3)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255])  # Default blue
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image, alpha=0.6)

def show_image_with_mask(image, mask, output_path):
    # Prepare figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert image to RGB for display
    show_mask(mask, ax)
    ax.axis('off')
    plt.tight_layout()

    # Save the visualization
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# Save image with bbox and keypoints overlay
def save_bbox_and_keypoints_overlay(image, bbox, keypoints, output_path):
    # Draw bounding box on a copy of the image
    image_with_annotations = image.copy()
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image_with_annotations, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

    # Draw visible keypoints
    for x, y in keypoints:
        cv2.circle(image_with_annotations, (int(x), int(y)), 5, (0, 0, 255), -1)  # Red dots

    # Save the result
    cv2.imwrite(output_path, image_with_annotations)


# Predict function
def predict_images_with_keypoints(image_folder, json_path, output_folder):
    mask_output_folder = os.path.join(output_folder, "mask_overlay")
    bbox_output_folder = os.path.join(output_folder, "bbox_overlay")

    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)
    if not os.path.exists(bbox_output_folder):
        os.makedirs(bbox_output_folder)

    # Initialize predictor
    predictor = SAM2ImagePredictor(sam2_model)

    # Load annotations
    annotations = load_annotations(json_path)
    print("Annotations Loaded:", annotations)

    # Process only images listed in the JSON file
    for file_name, annotation in annotations.items():
        # List to store prediction times
        prediction_times = []
        start_time = time.time()  # Start timer

        image_path = os.path.join(image_folder, file_name)

        # Load image
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Error loading image: {file_name}")

            # Ensure the image dimensions are valid
            if image.shape[0] <= 0 or image.shape[1] <= 0:
                raise ValueError(f"Invalid image dimensions for file: {file_name}")

            print(f"Processing {file_name} with bbox and keypoints")
        except Exception as e:
            print(f"Skipping {file_name} due to error: {e}")
            continue

        bbox = annotation['bbox']
        keypoints = process_keypoints(annotation['keypoints'])
        input_box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])

        # Perform prediction
        try:
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(
                point_coords=np.array(keypoints),
                point_labels=np.ones(len(keypoints)),
                box=input_box[None, :],
                multimask_output=False,
            )


            if masks is None or len(masks) == 0:
                raise ValueError(f"No prediction output for {file_name}")

            # Save mask overlay
            mask = masks[0]
            mask_output_path = os.path.join(mask_output_folder, f"{os.path.splitext(file_name)[0]}_mask_overlay.png")
            show_image_with_mask(image, mask, mask_output_path)

            # Save bbox and keypoints overlay
            bbox_output_path = os.path.join(bbox_output_folder, f"{os.path.splitext(file_name)[0]}_bbox_overlay.png")
            save_bbox_and_keypoints_overlay(image, bbox, keypoints, bbox_output_path)

            print(f"Mask and BBox visualizations saved for image: {file_name}")
        except Exception as e:
            print(f"Error during prediction for {file_name}: {e}")
        
        
        end_time = time.time()  # End timer
        prediction_time = end_time - start_time
        prediction_times.append(prediction_time)
        print(f"Prediction time for {file_name}: {prediction_time:.2f} seconds")

    # Calculate and print the average prediction time
    if prediction_times:
        average_time = sum(prediction_times) / len(prediction_times)
        print(f"Average prediction time: {average_time:.2f} seconds")



if __name__ == "__main__":
    # Specify the path to your model checkpoint
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    predictor = SAM2ImagePredictor(sam2_model)

    image_folder = "C:/Users/admin/AUTOAN/data/20240913_HumanGrayScale/annotations/yolonas_22keys/val_images"  # Specify the folder containing images
    json_path = "C:/Users/admin/AUTOAN/data/20240913_HumanGrayScale/annotations/yolonas_22keys/val_keypoints.json"  # Specify the JSON file path
    output_folder = "C:/Users/admin/sam2/results"  # Specify where to save predictions

    predict_images_with_keypoints(image_folder, json_path, output_folder)

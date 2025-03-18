import torch
import torch.nn as nn
import torchvision.transforms as transforms
import openslide
import numpy as np
import os
import pickle
from torchvision import models
from tqdm import tqdm

# Load trained model
model_path = "histology_models/best_model.pth"
device = torch.device("mps")

model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Binary classification (tumor vs. non-tumor)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)["model_state_dict"])
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define paths
test_images_dir = "/Volumes/BurhanAnisExtDrive/camelyon/camelyon_data/test/images"
tile_coords_file = "/Users/burhananis/fully-supervised-camelyon/data/tile_coords_test.pkl"
output_results_dir = "results/"
os.makedirs(output_results_dir, exist_ok=True)

# Load tile coordinates
with open(tile_coords_file, 'rb') as f:
    tile_coords = pickle.load(f)

# Get list of test slides
test_slide_names = sorted([f for f in os.listdir(test_images_dir) if f.endswith(".tif")])


# Allow user to specify number of slides to process
num_test_slides = int(input("Enter the number of slides to process: "))
if num_test_slides > 0:
    test_slide_names = test_slide_names[:num_test_slides]

# Check if previous results exist
tile_predictions_file = os.path.join(output_results_dir, "tile_predictions.pkl")
slide_predictions_file = os.path.join(output_results_dir, "slide_predictions.pkl")

# Load previous results if they exist
if os.path.exists(tile_predictions_file):
    with open(tile_predictions_file, "rb") as f:
        tile_predictions_dict = pickle.load(f)
else:
    tile_predictions_dict = {}

if os.path.exists(slide_predictions_file):
    with open(slide_predictions_file, "rb") as f:
        slide_predictions_dict = pickle.load(f)
else:
    slide_predictions_dict = {}

tile_size = 256

# Process test images
for slide_idx, slide_name in enumerate(tqdm(test_slide_names, desc="Processing Slides")):
    if slide_name in tile_predictions_dict:
        print(f"Skipping {slide_name}, already processed.")
        continue  # Skip slides that have already been processed

    slide_path = os.path.join(test_images_dir, slide_name)
    slide = openslide.OpenSlide(slide_path)
    slide_coords = tile_coords[slide_idx]  # Tile coordinates for this slide
    
    tile_predictions = []
    for (x, y) in tqdm(slide_coords, desc=f"Processing {os.path.basename(slide_name)}", leave=False):
        tile = slide.read_region((x, y), 0, (tile_size, tile_size)).convert("RGB")
        tile = transform(tile).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(tile)
            prediction = torch.argmax(output, dim=1).item()
        
        tile_predictions.append((x, y, prediction))
    
    # Save after processing each slide
    tile_predictions_dict[slide_name] = tile_predictions
    slide_predictions_dict[slide_name] = max(pred[2] for pred in tile_predictions)

    with open(tile_predictions_file, "wb") as f:
        pickle.dump(tile_predictions_dict, f)

    with open(slide_predictions_file, "wb") as f:
        pickle.dump(slide_predictions_dict, f)

    print(f"Saved predictions for {slide_name}.")

print(f"Inference complete. Predictions saved in {output_results_dir}")


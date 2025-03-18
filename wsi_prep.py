import openslide
import numpy as np
from scipy.ndimage import sobel
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


def get_tissue_mask(path, level=4, plot=False):
    """Takes image and generates binary tissue mask using HSV saturation channel."""
    slide = openslide.OpenSlide(path)
    mask_dims = slide.level_dimensions[level]
    mask_width, mask_height = mask_dims
    mask_image = slide.read_region((0, 0), level, mask_dims)
    mask_np = np.array(mask_image, copy=False)
    
    if mask_np.shape[2] > 3: #get rid of alpha channel, if present
        mask_np = mask_np[:, :, :3]
    
    # Convert RGB to HSV color space
    hsv_mask = cv2.cvtColor(mask_np, cv2.COLOR_RGB2HSV)
    
    # Extract the saturation channel
    saturation = hsv_mask[:, :, 1]
    
    # Threshold to create a tissue mask
    threshold_value = threshold_otsu(saturation)
    tissue_mask = saturation > threshold_value
    
    slide.close()
    
    if plot:
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(mask_np)
        ax[0].axis('off')
        ax[0].set_title('Original Image')
        
        ax[1].imshow(saturation, cmap='gray')
        ax[1].axis('off')
        ax[1].set_title('Saturation Channel')
        
        ax[2].imshow(tissue_mask, cmap='gray')
        ax[2].axis('off')
        ax[2].set_title('Tissue Mask')
    
    return tissue_mask



def get_tiles(path, tissue_mask, mask_level = 4, patch_size_level0 = 256, plot = False):

    """Tiles image on target level based on tissue mask then scales coordinates to level 0"""
    slide = openslide.OpenSlide(path)
    #get correct patch size
    downsample_factor = slide.level_downsamples[mask_level]
    patch_size = int(patch_size_level0 / downsample_factor)

    positions = []
    step_size = patch_size  # Non-overlapping patches
    for y in range(0, tissue_mask.shape[0] - patch_size + 1, step_size):
        for x in range(0, tissue_mask.shape[1] - patch_size + 1, step_size):
            patch = tissue_mask[y:y+patch_size, x:x+patch_size]
            # Check if the patch contains sufficient tissue
            if np.sum(patch) > (0.02 * patch_size * patch_size):  # Adjust threshold as needed for your images
                positions.append((x, y))

    # Convert positions to original image coordinates
    positions_level0 = [(int(x * downsample_factor), int(y * downsample_factor)) for (x, y) in positions]

    if plot:
        level = slide.level_count - 1
        width, height = slide.level_dimensions[level]
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(np.array(slide.read_region([0,0], level, slide.level_dimensions[level])))
        ax.axis('off')
        downsample_factor = slide.level_downsamples[level]

        # Overlay tiles
        for coord in positions_level0:
            x, y = coord
            # Scale coordinates to the specified level
            x_level = int(x / downsample_factor)
            y_level = int(y / downsample_factor)

            # Add a rectangle to the plot
            rect = patches.Rectangle(
                (x_level, y_level),  # Top-left corner
                patch_size_level0 / downsample_factor,  # Width
                patch_size_level0 / downsample_factor,  # Height
                linewidth=1.5,
                edgecolor='black',
                facecolor='none'
            )
            ax.add_patch(rect)

        plt.show()

    return positions_level0
import argparse
import sys
import cv2
import numpy as np
import openslide

def find_optimal_threshold(image_path):
    try:
        # Load a sample image
        slide = openslide.OpenSlide(image_path)
        thumbnail = slide.get_thumbnail([1024, 1024])  # Sample region
        thumbnail_np = np.array(thumbnail)[:, :, 0:3]
        thumbnail_bgr = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2BGR)
        hsv_image = cv2.cvtColor(thumbnail_bgr, cv2.COLOR_BGR2HSV)
        
        # Create window with trackbar
        cv2.namedWindow('Threshold Adjustment')
        
        def on_trackbar(val):
            s_channel = hsv_image[:, :, 1]
            _, binary_mask = cv2.threshold(s_channel, val, 255, cv2.THRESH_BINARY)
            result = cv2.bitwise_and(thumbnail_bgr, thumbnail_bgr, mask=binary_mask)
            cv2.imshow('Threshold Adjustment', result)
        
        # Create trackbar
        cv2.createTrackbar('S Threshold', 'Threshold Adjustment', 30, 255, on_trackbar)
        
        # Initialize display
        on_trackbar(30)
        
        print("Adjust the threshold using the trackbar.")
        print("Press any key to exit when you've found the optimal threshold.")
        
        cv2.waitKey(0)
        
        # Get the final threshold value
        final_threshold = cv2.getTrackbarPos('S Threshold', 'Threshold Adjustment')
        print(f"Selected threshold value: {final_threshold}")
        
        cv2.destroyAllWindows()
        return final_threshold
        
    except Exception as e:
        print(f"Error processing slide: {str(e)}", file=sys.stderr)
        return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Find optimal threshold for slide segmentation')
    parser.add_argument('slide_path', type=str, help='Path to the slide file')
    parser.add_argument('--output', '-o', type=str, help='Output file to save the threshold value (optional)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the slide
    print(f"Processing slide: {args.slide_path}")
    threshold = find_optimal_threshold(args.slide_path)
    
    if threshold is not None:
        # Save threshold to output file if specified
        if args.output:
            try:
                with open(args.output, 'w') as f:
                    f.write(str(threshold))
                print(f"Threshold value {threshold} saved to {args.output}")
            except Exception as e:
                print(f"Error saving threshold to file: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    main()

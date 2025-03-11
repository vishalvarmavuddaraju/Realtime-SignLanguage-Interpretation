import os
import numpy as np
import cv2

# Base directory for gestures
gesture_dir = "gestures"

# Automatically detect existing gesture folders
gesture_folders = sorted([f for f in os.listdir(gesture_dir) if os.path.isdir(os.path.join(gesture_dir, f))])

# Load gesture images from available folders
def load_gesture_images():
    img_list = []
    for folder in gesture_folders:
        img_path = os.path.join(gesture_dir, folder, "100.jpg")  # Pick a sample image
        if os.path.exists(img_path):
            img = cv2.imread(img_path, 0)
            if img is not None:
                img_list.append(img)
            else:
                print(f"⚠ Warning: {img_path} could not be loaded.")
        else:
            print(f"⚠ Warning: {img_path} does not exist.")
    return img_list

# Stack images vertically while ensuring they match in width
def stack_images(img_list):
    if not img_list:
        return np.zeros((50, 50), dtype=np.uint8)  # Return blank image if no valid images exist
    
    min_width = min(img.shape[1] for img in img_list)  # Find smallest width
    img_list = [cv2.resize(img, (min_width, img.shape[0])) for img in img_list]  # Resize to match widths
    
    return np.vstack(img_list)

# Main execution
gesture_images = load_gesture_images()
full_img = stack_images(gesture_images)

cv2.imshow("Stacked Gestures", full_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

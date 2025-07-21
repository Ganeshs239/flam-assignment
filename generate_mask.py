import cv2
import numpy as np
from PIL import Image
from rembg import remove
import io

def remove_bg_and_generate_mask(input_image_path, output_image_path, output_mask_path):
    """
    Removes background, saves transparent PNG, and generates a binary mask.
    
    Args:
        input_image_path (str): Path to input image.
        output_image_path (str): Where to save the transparent PNG.
        output_mask_path (str): Where to save the mask (white=object, black=bg).
    """
    # Step 1: Remove background using rembg (AI)
    with open(input_image_path, "rb") as f:
        img_bytes = f.read()
    
    # Remove background (returns transparent PNG as bytes)
    img_no_bg_bytes = remove(img_bytes)
    img_no_bg = Image.open(io.BytesIO(img_no_bg_bytes))
    
    # Save transparent PNG
    img_no_bg.save(output_image_path)
    
    # Step 2: Generate mask from transparency
    img_no_bg_arr = np.array(img_no_bg)
    mask = (img_no_bg_arr[:, :, 3] > 0).astype(np.uint8) * 255  # Alpha channel -> binary mask
    
    # Optional: Refine mask edges with OpenCV
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    
    # Save mask
    cv2.imwrite(output_mask_path, mask)
    print(f"Saved transparent image to: {output_image_path}")
    print(f"Saved mask to: {output_mask_path}")

# Example usage
input_image = "input/person.jpg"          # Your input image
output_image = "input/obj.png"        # Output (transparent PNG)
output_mask = "input/obj_mask.png"           # Output (binary mask)

remove_bg_and_generate_mask(input_image, output_image, output_mask)
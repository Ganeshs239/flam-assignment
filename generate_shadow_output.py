import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import torch.nn.functional as F
# from get_light_direction_from_shadow import get_light_direction_from_shadow
from location import detect_brightest_direction


def estimate_depth(img):
    """Estimate depth using MiDaS model"""
    try:
        # Load MiDaS model
        model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        model.eval()
        
        # MiDaS transform
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.dpt_transform
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Apply transforms and predict
        input_tensor = transform(img_pil).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(input_tensor)
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        return depth_map
        
    except Exception as e:
        print(f"Error with MiDaS depth estimation: {e}")
        # Fallback: create a simple gradient depth map
        h, w = img.shape[:2]
        depth_map = np.linspace(0, 1, h).reshape(-1, 1).repeat(w, axis=1)
        return depth_map


def create_realistic_shadow(obj_mask, depth_map, light_direction=(1, -1), shadow_intensity=0.4, blur_radius=25):
    """Create a more realistic shadow based on object mask and depth"""
    h, w = obj_mask.shape[:2]
    
    # Normalize mask
    if obj_mask.max() > 1:
        obj_mask = obj_mask.astype(np.float32) / 255.0
    
    # Create shadow displacement based on light direction and depth
    # offset_x = int(light_direction[0] * 20 * (1 - depth_map.mean()))
    # offset_y = int(light_direction[1] * 20 * (1 - depth_map.mean()))

    shadow_length = 60  # Strong long shadows
    offset_x = int(light_direction[0] * shadow_length)
    offset_y = int(light_direction[1] * shadow_length)  # was *20

    
    # Create shadow by shifting the object mask
    shadow_mask = np.zeros_like(obj_mask)
    
    # Handle positive and negative offsets
    if offset_x >= 0 and offset_y >= 0:
        shadow_mask[offset_y:, offset_x:] = obj_mask[:h-offset_y, :w-offset_x]
    elif offset_x >= 0 and offset_y < 0:
        shadow_mask[:h+offset_y, offset_x:] = obj_mask[-offset_y:, :w-offset_x]
    elif offset_x < 0 and offset_y >= 0:
        shadow_mask[offset_y:, :w+offset_x] = obj_mask[:h-offset_y, -offset_x:]
    else:  # both negative
        shadow_mask[:h+offset_y, :w+offset_x] = obj_mask[-offset_y:, -offset_x:]

    # Add elliptical shadow at feet for grounding effect
    # # Find bottom center of object mask
    coords = np.column_stack(np.where(obj_mask > 0))
    if coords.size > 0:
        bottom = coords[:, 0].max()
        row_indices = coords[coords[:, 0] == bottom]
        if row_indices.size > 0:
            center_x = int(np.mean(row_indices[:, 1]))
            center_y = bottom + 3  # slightly below the foot
            cv2.ellipse(shadow_mask, (center_x, center_y), (15, 6), 0, 0, 360, 1.0, -1)

    
    # Remove shadow where object exists (shadows don't appear under the object)
    shadow_mask = shadow_mask * (1 - obj_mask)
    
    # Apply depth-based shadow intensity (closer objects cast darker shadows)
    shadow_strength = shadow_intensity * (1 - depth_map * 0.5)
    shadow_mask = shadow_mask * shadow_strength
    
    # Blur shadow for realism
    if blur_radius > 0:
        shadow_mask = cv2.GaussianBlur(shadow_mask, (blur_radius*2+1, blur_radius*2+1), blur_radius/3)
    
    return shadow_mask


def enhanced_blend_object_shadow(bg, obj, obj_mask, light_direction=(1, -1), shadow_intensity=0.4):
    h_bg, w_bg = bg.shape[:2]
    h_obj, w_obj = obj.shape[:2]

    # Place object at desired location
    x_offset, y_offset = 100, h_bg - h_obj - 20  # Adjust manually

    # Prepare a blank mask for the full background size
    full_mask = np.zeros((h_bg, w_bg), dtype=np.uint8)
    full_obj = np.zeros_like(bg)

    full_mask[y_offset:y_offset+h_obj, x_offset:x_offset+w_obj] = obj_mask
    full_obj[y_offset:y_offset+h_obj, x_offset:x_offset+w_obj] = obj

    # Estimate depth
    depth_map = estimate_depth(bg)

    # Shadow creation
    shadow_mask = create_realistic_shadow(full_mask, depth_map, light_direction, shadow_intensity)

    # Shadow blending
    bg_float = bg.astype(np.float32) / 255.0
    shadow_mask_3ch = np.stack([shadow_mask]*3, axis=-1)
    shadowed_bg = bg_float * (1 - shadow_mask_3ch * 0.7)

    # Optional tint
    shadow_color = np.array([0.8, 0.85, 0.9])
    shadowed_bg = shadowed_bg * (1 - shadow_mask_3ch) + \
                  (shadowed_bg * shadow_color) * shadow_mask_3ch

    # Blend object
    full_mask_3ch = np.stack([full_mask/255.0]*3, axis=-1)
    obj_float = full_obj.astype(np.float32) / 255.0
    result = shadowed_bg * (1 - full_mask_3ch) + obj_float * full_mask_3ch

    return (result * 255).astype(np.uint8)



def adjust_object_lighting(obj, obj_mask, light_direction=(1, -1), light_intensity=0.3):
    """Adjust object lighting to match the scene"""
    
    if obj_mask.max() > 1:
        obj_mask = obj_mask.astype(np.float32) / 255.0
    if len(obj_mask.shape) == 3:
        obj_mask = cv2.cvtColor(obj_mask, cv2.COLOR_BGR2GRAY)
    
    obj_float = obj.astype(np.float32) / 255.0
    h, w = obj.shape[:2]
    
    # Create lighting gradient
    x_grad = np.linspace(-1, 1, w)
    y_grad = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x_grad, y_grad)
    
    # Calculate lighting based on light direction
    lighting = (X * light_direction[0] + Y * light_direction[1]) * light_intensity
    lighting = np.clip(lighting + 1, 0.5, 1.5)  # Normalize to reasonable range
    
    # Apply lighting only to the object
    lighting_3ch = np.stack([lighting, lighting, lighting], axis=-1)
    obj_mask_3ch = np.stack([obj_mask, obj_mask, obj_mask], axis=-1)
    
    lit_obj = obj_float * lighting_3ch * obj_mask_3ch + obj_float * (1 - obj_mask_3ch)
    
    return (np.clip(lit_obj, 0, 1) * 255).astype(np.uint8)


if __name__ == '__main__':
    print("Loading images...")
    
    # Load images
    bg_img = cv2.imread("BackG.jpg")
    obj_img = cv2.imread("input/person.jpg")
    obj_mask = cv2.imread("input/obj_mask.png", 0)
    
    if bg_img is None or obj_img is None or obj_mask is None:
        print("Error: Could not load one or more images. Please check file paths.")
        exit(1)
    
    print(f"Background shape: {bg_img.shape}")
    print(f"Object shape: {obj_img.shape}")
    print(f"Mask shape: {obj_mask.shape}")
    
    # Adjust object lighting to match scene
    print("Adjusting object lighting...")
    # light_direction = get_light_direction_from_shadow("bg.jpg")
    light = detect_brightest_direction(bg_img) 
    light_direction = (light[1],light[0])
    # Light coming from top-right
    obj_img_lit = adjust_object_lighting(obj_img, obj_mask, light_direction, light_intensity=0.2)
    
    # Create final composition with realistic shadows
    print("Compositing final image...")
    result = enhanced_blend_object_shadow(
        bg_img, 
        obj_img_lit, 
        obj_mask, 
        light_direction=light_direction,
        shadow_intensity=1.5
    )
    
    # Save result
    output_path = "enhanced_shadow_result.jpg"
    cv2.imwrite(output_path, result)
    print(f"✅ Enhanced output saved to {output_path}")
    
    # Optional: Create comparison image
    h, w = bg_img.shape[:2]
    comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
    comparison[:, :w] = cv2.resize(bg_img, (w, h))
    comparison[:, w:] = result
    
    cv2.imwrite("comparison.jpg", comparison)
    print("✅ Comparison image saved to comparison.jpg")
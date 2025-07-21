import cv2
import numpy as np

def detect_brightest_direction(image):
    """
    Estimate light direction by locating the brightest region in the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Find the location of the brightest pixel
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
    print(f"[INFO] Brightest point: {max_loc} with intensity {max_val}")

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    dx = max_loc[0] - center[0]
    dy = max_loc[1] - center[1]

    # Normalize direction
    magnitude = np.sqrt(dx**2 + dy**2)
    if magnitude == 0:
        return (0, 0), "Unknown"

    direction = (dx / magnitude, dy / magnitude)

    # Determine source label
    angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi
    if -45 <= angle <= 45:
        label = "Right"
    elif 45 < angle <= 135:
        label = "Bottom"
    elif angle > 135 or angle < -135:
        label = "Left"
    else:
        label = "Top"

    print(f"[INFO] Light appears to come from {label} → Direction vector: ({direction[0]:.2f}, {direction[1]:.2f})")
    return direction

def main():
    image_path = "image.jpg"  # Your uploaded file
    image = cv2.imread(image_path)

    if image is None:
        print("[ERROR] Failed to load image.")
        return

    direction, label = detect_brightest_direction(image)
    # result_img = draw_light_arrow(image, direction, label)

    # cv2.imshow("Detected Light Direction", result_img)
    # cv2.imwrite("output_brightest_light_arrow.jpg", result_img)
    # print("[INFO] Output saved as 'output_brightest_light_arrow.jpg'")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

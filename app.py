import cv2
import numpy as np

# Load the image
image_path = "output/lines.jpg"
image = cv2.imread(image_path)

# The coordinates of the four corners in clockwise order
src_points = np.array([
    [116, 221],  # Top-left
    [486, 217],  # Top-right
    [483, 727],  # Bottom-right
    [125, 728]   # Bottom-left
], dtype=np.float32)

# Compute the width and height of the bird's-eye view rectangle
width = int(max(
    np.linalg.norm(src_points[1] - src_points[0]),  # Top edge
    np.linalg.norm(src_points[2] - src_points[3])   # Bottom edge
))
height = int(max(
    np.linalg.norm(src_points[0] - src_points[3]),  # Left edge
    np.linalg.norm(src_points[1] - src_points[2])   # Right edge
))

# Define the destination points for the perspective transform
dst_points = np.array([
    [0, 0],               # Top-left
    [width - 1, 0],       # Top-right
    [width - 1, height - 1],  # Bottom-right
    [0, height - 1]       # Bottom-left
], dtype=np.float32)

# Compute the perspective transform matrix
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Perform the perspective warp
warped_image = cv2.warpPerspective(image, matrix, (width, height))

# Save the result
output_path = "warped_image.png"
cv2.imwrite(output_path, warped_image)

# Optional: Display the result (if running locally)
# cv2.imshow("Warped Image", warped_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(f"Perspective-transformed image saved to: {output_path}")

import cv2
import numpy as np

# Load the image
image = cv2.imread("assets/Scan_0011.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 1: Apply Thresholding to create a binary image
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# Step 2: Use morphological operations to isolate vertical lines
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))  # Tall kernel for vertical lines
detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

# Step 3: Find contours of the vertical lines
contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Step 4: Filter and draw bounding boxes around 40-pixel-wide columns
output_image = image.copy()

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    # Filter for approximately 40-pixel wide lines (adjust tolerance as needed)
    if 35 <= w <= 45:  # Width near 40 pixels
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show the output image
cv2.imshow("Detected Columns", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

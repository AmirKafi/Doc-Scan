import cv2
import numpy as np

# Load the image
image = cv2.imread('assets/Scan_0011.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 1: Edge detection (Canny edge detection)
edges = cv2.Canny(gray, 50, 150)

# Step 2: Find contours for perspective correction (skew correction)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Assume the largest contour is the document
largest_contour = max(contours, key=cv2.contourArea)

# Step 3: Perspective transformation (if needed)
# Get the bounding box and apply perspective correction here

# Step 4: Increase contrast using histogram equalization
equalized = cv2.equalizeHist(gray)

# Step 5: Apply sharpening (Unsharp mask)
blurred = cv2.GaussianBlur(equalized, (5, 5), 10.0)

# Step 6: Thresholding for binarization
_, binarized = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

# Step 7: Noise reduction (using median filter)
denoised = cv2.medianBlur(binarized, 3)

# Show the final output
cv2.imshow('Processed Image', denoised)
cv2.waitKey(0)
cv2.destroyAllWindows()

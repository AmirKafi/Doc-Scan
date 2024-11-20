import cv2
import numpy as np

# Step 1: Read the image
image = cv2.imread('photo_2024-11-20_00-56-20.jpg')


# Step 2: Preprocess the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 80, 200)

def CannyThreshold(val):
    low_threshold = val
    detected_edges = cv2.Canny(blurred, low_threshold, low_threshold * 2)
    mask = detected_edges != 0
    dst = image * (mask[:,:,None].astype(image.dtype))
    cv2.imshow('Edge Map', dst)

cv2.namedWindow('Edge Map', cv2.WINDOW_NORMAL)
cv2.createTrackbar('track', 'Edge Map', 0, 200, CannyThreshold)
CannyThreshold(0)

# Step 3: Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
cv2.imshow('Contour Image', contour_image)

# Step 4: Identify the largest contour that looks like a document
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(approx) == 4:  # Check if it's a quadrilateral
        doc_contour = approx
        break

# Step 5: Draw the contour and corner points
if 'doc_contour' in locals():
    points = doc_contour.reshape(4, 2)  # Extract the corner points
    for i, point in enumerate(points):
        # Draw circles at each corner
        cv2.circle(image, tuple(point), 10, (0, 255, 0), -1)
        # Label the points
        cv2.putText(image, f'P{i+1}', tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    print("Corner points:", points)
    cv2.imshow("Detected Document with Points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Document not found!")

import cv2
import numpy as np

# Load the image
image = cv2.imread("assets/Scan_0011.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Preprocessing: Adaptive thresholding
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Get the image dimensions
height, width = binary.shape

# Define left and right regions (shrinking the detection area)
left_region = binary[:, :int(width * 0.1)]  # Left 10% of the image
right_region = binary[:, int(width * 0.9):]  # Right 10% of the image

# Create a full-sized black image (with padding)
full_image = np.zeros_like(binary)

# Place the left and right regions into the full-sized image
full_image[:, :int(width * 0.1)] = left_region  # Place left region into the full image
full_image[:, int(width * 0.9):] = right_region  # Place right region into the full image

# Find contours in the full image (which now has the left and right regions)
contours, _ = cv2.findContours(full_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours to detect rectangles within the left and right regions
rectangles = []
for contour in contours:
    # Approximate the contour
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

    # Check if it's a rectangle (4 corners) and is of reasonable size
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)

        # Filter by aspect ratio and size (adjust thresholds as needed)
        aspect_ratio = w / float(h)
        if 0.1 < aspect_ratio < 10 < w < 100 and 10 < h < 100:  # Adjust size filter as needed
            # Filter by area to exclude very small contours like QR codes
            area = cv2.contourArea(contour)
            if area > 500:  # Adjust based on the size of the rectangles you want to keep
                # Now, the positions are in the full image context, so we can keep them
                rectangles.append((x, y, w, h))

# Sort the rectangles by their vertical position (y-coordinate)
rectangles = sorted(rectangles, key=lambda x: x[1])  # Sort by top y coordinate

# Draw the rectangles on the original image
output_image = image.copy()
for (x, y, w, h) in rectangles:
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Now, we will draw parallel lines from the top and bottom of each rectangle
line_pairs = []
for i in range(len(rectangles)):
    left_x, left_y, left_w, left_h = rectangles[i]

    # Find a matching rectangle in the right region
    for j in range(i + 1, len(rectangles)):
        right_x, right_y, right_w, right_h = rectangles[j]

        # Match rectangles based on their vertical position (y-coordinate) and check if they are at the same level
        if abs(left_y - right_y) < 10 and abs((left_y + left_h) - (right_y + right_h)) < 10:
            # Top line: from the top-left of the left rectangle to the top-right of the right rectangle
            top_line = ((left_x + left_w, left_y), (right_x, right_y))
            # Bottom line: from the bottom-left of the left rectangle to the bottom-right of the right rectangle
            bottom_line = ((left_x + left_w, left_y + left_h), (right_x, right_y + right_h))

            line_pairs.append((top_line, bottom_line))

# Draw the parallel lines (top and bottom) for visualization
for top_line, bottom_line in line_pairs:
    cv2.line(output_image, top_line[0], top_line[1], (0, 0, 255), 2)  # Top line in red
    cv2.line(output_image, bottom_line[0], bottom_line[1], (0, 0, 255), 2)  # Bottom line in red

# Now we detect rounded rectangles (columns) within the lines' regions
for top_line, bottom_line in line_pairs:
    # Extract the bounding box between the lines
    top_left = top_line[0]
    bottom_right = bottom_line[1]

    # Define the region of interest (ROI) between the two lines
    roi = binary[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # Find contours in this region
    roi_contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for roi_contour in roi_contours:
        # Approximate the contour to get a polygonal shape
        approx = cv2.approxPolyDP(roi_contour, 0.02 * cv2.arcLength(roi_contour, True), True)

        # Filter out small contours and those that don't resemble rounded rectangles
        if len(approx) >= 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)

            # Allow more flexibility in size matching, not just exact dimensions
            if w > 10 and h > 10:  # Ensure the detected shape is reasonably large enough to be a column
                # Draw the detected rounded rectangle (column)
                cv2.rectangle(output_image, (top_left[0] + x, top_left[1] + y),
                              (top_left[0] + x + w, top_left[1] + y + h), (0, 0, 255), 2)

# Display the final output
cv2.namedWindow('Rounded Rectangles Detection', cv2.WINDOW_NORMAL)
cv2.imshow("Rounded Rectangles Detection", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

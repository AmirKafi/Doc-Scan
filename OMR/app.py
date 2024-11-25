import time

import cv2

from OMR.utils import find_rectangles, correct_rows_position, dump_result

start = time.time()

file_name = 'QalamChi'

# Load the image
image = cv2.imread(f"output/{file_name}.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Preprocessing: Adaptive thresholding
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Apply Gaussian Blur to reduce noise before edge detection
blurred = cv2.GaussianBlur(binary, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rectangles = find_rectangles(binary)

# Define the target size for the large rectangle (1430x1650 pixels)
target_width = 1430
target_height = 1650

detected_columns = []
# Loop through contours to find the large rectangle
for contour in contours:
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        if abs(w - target_width) < 100 and abs(h - target_height) < 100:
            answer_block_row_count = 5
            answer_block_col_count = 6
            answer_block_width = w // answer_block_col_count
            answer_block_height = h // answer_block_row_count
            for i in range(answer_block_row_count):
                for j in range(answer_block_col_count):
                    x1 = x + j * answer_block_width
                    y1 = y + i * answer_block_height
                    x2 = x1 + answer_block_width
                    y2 = y1 + answer_block_height
                    column_width = 37
                    gap_width = 10
                    num_columns = (answer_block_width + gap_width) // (column_width + gap_width)
                    for col in range(1, num_columns):
                        col_x1 = x1 + col * (column_width + gap_width)
                        col_x2 = col_x1 + column_width
                        detected_columns.append((col_x1, 0, col_x2, 0))

answered_question, image = correct_rows_position(rectangles, detected_columns, image)

# Dump The results in json
dump_result(answered_question,file_name)

# Show the resulting image with highlighted matches
cv2.namedWindow('Highlighted Rows and Matches', cv2.WINDOW_NORMAL)
cv2.imshow("Highlighted Rows and Matches", image)
end = time.time()
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Step took {end - start:.2f} seconds")

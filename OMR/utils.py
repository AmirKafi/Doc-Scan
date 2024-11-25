import json
from typing import Set, Tuple, Sequence, Any, List

import cv2
import numpy as np
from numpy import ndarray


def find_rectangles(binary):
    # Get the image dimensions
    height, width = binary.shape

    # Define left and right regions (shrinking the detection area)
    left_region = binary[:, :int(width * 0.06)]  # Left 6% of the image
    right_region = binary[:, int(width * 0.94):]  # Right 6% of the image

    # Create a full-sized black image (with padding)
    full_image = np.zeros_like(binary)

    # Place the left and right regions into the full-sized image
    full_image[:, :int(width * 0.06)] = left_region  # Place left region into the full image
    full_image[:, int(width * 0.94):] = right_region  # Place right region into the full image

    # Find contours in the full image (which now has the left and right regions)
    contours_rec, _ = cv2.findContours(full_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours to detect rectangles within the left and right regions
    rectangles = []
    for contour in contours_rec:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.1 < aspect_ratio < 10 < w < 100 and 10 < h < 100:
                area = cv2.contourArea(contour)
                if area > 500:
                    rectangles.append((x, y, w, h))

    # Sort the rectangles by their vertical position (y-coordinate)
    rectangles = sorted(rectangles, key=lambda x: x[1])

    rectangles = rectangles[20:120]

    rects = pair_rectangles(rectangles, width)
    return rects


def pair_rectangles(rectangles, image_width):
    # Separate rectangles into left and right based on x-coordinates
    left_rectangles = [rect for rect in rectangles if rect[0] < image_width * 0.06]
    right_rectangles = [rect for rect in rectangles if rect[0] > image_width * 0.94]

    return list(zip(left_rectangles, right_rectangles))


def detect_rows(contours: Sequence[ndarray | Any]):
    # Define the target size for the large rectangle (1430x1650 pixels)
    target_width = 1430
    target_height = 1650
    # Initialize a list for detected rows within columns
    detected_rows = []
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
                            margin_top = 10
                            row_start_y = y1 + margin_top
                            row_height = 17
                            row_gap = 14
                            group_gap = 50
                            rows_per_group = 10
                            for group in range(3):
                                group_start_y = row_start_y + group * (
                                        rows_per_group * (row_height + row_gap) + group_gap)
                                for row in range(rows_per_group):
                                    row_y1 = group_start_y + row * (row_height + row_gap)
                                    row_y2 = row_y1 + row_height
                                    if row_y2 <= y2:
                                        detected_rows.append((col_x1, row_y1, col_x2, row_y2))

        return detected_rows


def correct_rows_position(rects, rows: list[Tuple[int, int, int, int]], image):
    # Map rows to their corresponding side rectangles
    corrected_columns: Set[Tuple[int, int, int, int]] = set()
    for rect in rects:
        lx, ly, lw, lh = rect[0]
        rx, ry, rw, rh = rect[1]

        left_rect_y1 = ly
        left_rect_y2 = ry + rh

        right_rect_y1 = ly
        right_rect_y2 = ry + rh

        for (colX1, rowY1, colX2, rowY2) in rows:
            rect_y1 = int((left_rect_y1 + right_rect_y1) / 2)
            rect_y2 = int((left_rect_y2 + right_rect_y2) / 2)

            corrected_columns.add((colX1, rect_y1, colX2, rect_y2))

    image = get_answered(sorted(corrected_columns), image)
    return image


import cv2
from typing import List, Tuple

import cv2
from typing import List, Tuple


def get_answered(answers: List[Tuple[int, int, int, int]], image) -> tuple[list[tuple[int | Any, tuple[Any]]], Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    answered_questions = []
    temp_answers = []
    col = 0
    # Sort the answers by x-coordinate first (left to right)
    x_sorted_answers = sorted(answers, key=lambda x: x[1])
    y_sorted_answers = sorted(answers, key=lambda x: x[0])
    for index, answer in enumerate(x_sorted_answers):
        x1, y1, x2, y2 = answer
        row = y_sorted_answers.index(answer) % 50 + 1

        # Extract the Region of Interest (ROI)
        ROI = gray[y1:y2, x1:x2]

        # Threshold the ROI to identify filled areas
        _, thresh = cv2.threshold(ROI, 200, 255, cv2.THRESH_BINARY)

        # Calculate fill ratio
        total_pixels = thresh.size
        white_pixels = cv2.countNonZero(thresh)
        fill_ratio = white_pixels / total_pixels

        # Define threshold to determine if it's filled
        filled = fill_ratio < 0.5  # Adjust the threshold as needed


        if col == 6 and not temp_answers:
            col = 0
        if (index % 4) == 0:
            col += 1

        # Track the answers for the current question block
        temp_answers.append(filled)


        # When 4 answers have been collected (for one question), store them
        if len(temp_answers) == 4:
            question_number = ((col * 50) - 50) + row
            answered_questions.append((question_number,tuple(temp_answers)))
            temp_answers.clear()

        # Draw a rectangle on the original image for visualization
        color = (0, 255, 0) if filled else (0, 0, 255)  # Green if filled, Red otherwise
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    answered_questions = sorted(answered_questions, key=lambda x: x[0])
    return answered_questions,image

def dump_result(answered_questions: List[Tuple[int, Tuple[bool, bool, bool, bool]]],file_name:str):
    with open(f'output/{file_name}_Result.json', 'w') as file:
        json.dump(answered_questions, file, indent=4)

import json
from collections import OrderedDict
from typing import List, Tuple
from typing import Set, Any

import cv2
import numpy as np


def start_process(file_path: str):
    image = cv2.imread(file_path)

    # First Grayscale it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # then make it binary
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Then Blur it
    blurred = cv2.GaussianBlur(binary, (5, 5), 0)

    # And Then Create a Canny
    edges = cv2.Canny(blurred, 50, 150)

    # The Rectangles on the sides
    bounded_boxes = find_rectangles(binary)

    col_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    target_width = 1430
    target_height = 1650
    # Initialize a list for detected rows within columns
    detected_columns: Set[Tuple[int, int, int, int]] = set()
    # Loop through contours to find the large rectangle
    for col_contour in col_contours:
        epsilon = 0.04 * cv2.arcLength(col_contour, True)
        approx = cv2.approxPolyDP(col_contour, epsilon, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if abs(w - target_width) < 500 and abs(h - target_height) < 500:
                print(approx)
                answer_block_col_count = 6
                answer_block_row_count = 5
                answer_block_width = w // answer_block_col_count  # Width of each smaller box
                for i in range(answer_block_row_count):
                    for j in range(answer_block_col_count):
                        x1 = x + j * answer_block_width
                        column_width = answer_block_width / 6.5
                        gap_width = column_width / 3.5
                        num_columns = int((answer_block_width + gap_width) // (column_width + gap_width))
                        for col in range(1, num_columns):
                            col_x1 = int(x1 + col * (column_width + gap_width))
                            col_x2 = int(col_x1 + column_width)
                            for rect in bounded_boxes:
                                lx, ly, lw, lh = rect[0]
                                rx, ry, rw, rh = rect[1]

                                left_rect_y1 = ly
                                left_rect_y2 = ry + rh

                                right_rect_y1 = ly
                                right_rect_y2 = ry + rh

                                rect_y1 = int((left_rect_y1 + right_rect_y1) / 2)
                                rect_y2 = int((left_rect_y2 + right_rect_y2) / 2)

                                detected_columns.add((col_x1, rect_y1, col_x2, rect_y2))

    answered, image = get_answered(sorted(detected_columns), image)

    return answered, image


def find_rectangles(binary):
    # Get the image dimensions
    height, width = binary.shape

    # Define left and right regions (shrinking the detection area)
    left_region = binary[:, :int(width * 0.08)]  # Left 6% of the image
    right_region = binary[:, int(width * 0.92):]  # Right 6% of the image

    # Create a full-sized black image (with padding)
    full_image = np.zeros_like(binary)

    # Place the left and right regions into the full-sized image
    full_image[:, :int(width * 0.08)] = left_region  # Place left region into the full image
    full_image[:, int(width * 0.92):] = right_region  # Place right region into the full image

    cv2.imshow('full_image', full_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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


def get_answered(answers: List[Tuple[int, int, int, int]], image) -> tuple[dict[int | Any, dict[Any, Any]], Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    answered_questions = {}
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

        if thresh is not None:
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
            temp_answers.append((len(temp_answers) + 1, filled))

            # When 4 answers have been collected (for one question), store them
            if len(temp_answers) == 4:
                question_number = ((col * 50) - 50) + row
                if question_number not in answered_questions:
                    answered_questions[question_number] = {}

                for answer_number, filled in temp_answers:
                    answered_questions[question_number][str(answer_number)] = filled

                temp_answers.clear()

            # Draw a rectangle on the original image for visualization
            color = (0, 255, 0) if not filled else (0, 0, 255)  # Green if filled, Red otherwise
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        else:
            print(ROI)
            print(thresh)

    answered_questions = OrderedDict(sorted(answered_questions.items(), key=lambda x: x[0]))

    return answered_questions, image


def dump_result(answered_questions: List[Tuple[int, Tuple[bool, bool, bool, bool]]], file_name: str):
    with open(f'output/{file_name}_Result.json', 'w') as file:
        json.dump(answered_questions, file, indent=4)

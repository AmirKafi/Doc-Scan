from collections import OrderedDict
from math import floor, ceil
from typing import List, Tuple
from typing import Set, Any

import cv2

from Configs.config import ANSWER_COL_WIDTH, ANSWERS_COL_X_COORDINATE, DOC_WIDTH, DOC_HEIGHT, BOUNDED_BOX_MAX_AREA, \
    DIS_BETWEEN_BOUNDED_BOXES, DIS_TO_FIRST_BOUNDED_BOX


def start_process(raw_bird_eye_view_img):
    # First Grayscale it

    gray = cv2.cvtColor(raw_bird_eye_view_img, cv2.COLOR_BGR2GRAY)

    # then make it binary
    _, binary = cv2.threshold(gray, 0, 240, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # The Rectangles on the sides
    bounded_boxes, raw_bird_eye_view_img = find_rectangles(binary, raw_bird_eye_view_img)

    height, width = binary.shape[:2]

    # Initialize a list for detected rows within columns
    detected_columns: Set[Tuple[int, int, int, int]] = set()
    # Loop through contours to find the large rectangle
    for rect in bounded_boxes:
        lx, ly, lw, lh = rect[0]
        rx, ry, rw, rh = rect[1]

        for col_index,x_cor in enumerate(ANSWERS_COL_X_COORDINATE):

            cv2.rectangle(raw_bird_eye_view_img, (lx, ly), (lx + lw, ly + lh), (255,0,255), 1)
            cv2.rectangle(raw_bird_eye_view_img, (rx, ry), (rx + rw, ry + rh), (255,0,255), 1)

            if not width == DOC_WIDTH:
                dis_between_bounded_boxes = rx - (lx + lw)
                coef = dis_between_bounded_boxes / DIS_BETWEEN_BOUNDED_BOXES
                width_coef =width / DOC_WIDTH

                start = lx + lw

                x_cor = x_cor - DIS_TO_FIRST_BOUNDED_BOX
                start_x = x_cor
                end_x = x_cor + ANSWER_COL_WIDTH

                start_x = (start_x * width_coef)

                end_x = (end_x * width_coef)

                start_x = start_x + start

                end_x = end_x + start
            else:
                start_x = x_cor
                end_x = x_cor + ANSWER_COL_WIDTH


            top_y = ly + col_index * ((ry - ly) / len(ANSWERS_COL_X_COORDINATE))
            bottom_y = (ly + lh) + col_index * (((ry + rh) - (ly + lh)) / len(ANSWERS_COL_X_COORDINATE))


            detected_columns.add((int(start_x), int(top_y), int(end_x),int(bottom_y)))

    answered, image = get_answered(sorted(detected_columns), raw_bird_eye_view_img)

    return answered, image

def find_rectangles(binary, raw_bird_eye_view_img):
    full_image = binary
    height, width = binary.shape[:2]
    width_coefficient = width / DOC_WIDTH
    height_coefficient = height / DOC_HEIGHT
    coefficient = (width_coefficient + height_coefficient) / 2

    contours_rec, _ = cv2.findContours(full_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for contour in contours_rec:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        bounded_boxes = cv2.boundingRect(approx)
        x, y, w, h = bounded_boxes
        aspect_ratio = w / float(h)
        if aspect_ratio > 1:
            area = cv2.contourArea(contour)
            if (coefficient * BOUNDED_BOX_MAX_AREA) > area > 15:
                rectangles.append((x, y, w, h))

    rectangles = group_tuples_by_second_value(rectangles, 25)

    new_rects = []

    _, _, target_width, target_height = rectangles[0][0]

    for rec in rectangles:
        if len(rec) == 60:
            alignment_x = sum(rect[0] + rect[2] // 2 for rect in rec) // len(
                rec) - target_width // 2
            rec = resize_bounded_boxes(rec,target_width,target_height,alignment_x)
            print(rec)
            rec = sorted(rec, key=lambda y: y[1])
            rec = rec[10:]
            new_rects.append(rec)
    rectangles = list(zip(new_rects[1], new_rects[0]))

    return rectangles, raw_bird_eye_view_img


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
        _, thresh = cv2.threshold(ROI, 210, 210, cv2.THRESH_BINARY)

        if thresh is not None:
            total_pixels = thresh.size
            white_pixels = cv2.countNonZero(thresh)
            fill_ratio = white_pixels / total_pixels

            # Define threshold to determine if it's filled
            filled = fill_ratio < 0.7  # Adjust the threshold as needed

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
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        else:
            pass

    answered_questions = OrderedDict(sorted(answered_questions.items(), key=lambda x: x[0]))

    return answered_questions, image

def group_tuples_by_second_value(tuples, tolerance):
    groups = []

    for item in tuples:
        added = False
        for group in groups:
            if abs(group[0][0] - item[0]) <= tolerance:
                group.append(item)
                added = True
                break
        if not added:
            groups.append([item])

    return groups

def resize_bounded_boxes(bound_boxes,target_w,target_h,alignment_x):
    resized_bounded_boxes = []
    for box in bound_boxes:
        x, y, w, h = box
        x = int(alignment_x + (w - target_w) / 2)
        y = int(y + (h - target_h) / 2)
        resized_bounded_boxes.append((x, y, target_w, target_h))

    return resized_bounded_boxes
import json
import random
from collections import OrderedDict
from typing import List, Tuple
from typing import Set, Any

import cv2
import numpy as np

from Configs.config import ANSWER_COL_WIDTH, ANSWERS_COL_X_COORDINATE, DOC_WIDTH, DOC_HEIGHT, BOUNDED_BOX_MIN_AREA, \
    BOUNDED_BOX_MAX_AREA, DIS_BETWEEN_BOUNDED_BOXES


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
        for col_index,x_cor in enumerate(ANSWERS_COL_X_COORDINATE):
            lx, ly, lw, lh = rect[0]
            rx, ry, rw, rh = rect[1]

            cv2.rectangle(raw_bird_eye_view_img, (lx, ly), (lx + lw, ly + lh), (255,0,255), 1)
            cv2.rectangle(raw_bird_eye_view_img, (rx, ry), (rx + rw, ry + rh), (255,0,255), 1)

            width_coef = width / DOC_WIDTH

            start_x = (x_cor * width_coef)
            end_x = (x_cor * width_coef) + (ANSWER_COL_WIDTH * width_coef)

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
    for rec in rectangles:
        if len(rec) == 60:
            rec = sorted(rec, key=lambda rec: rec[1])
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
        _, thresh = cv2.threshold(ROI, 180, 255, cv2.THRESH_BINARY)

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


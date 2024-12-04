from collections import OrderedDict
from math import floor, ceil
from typing import List, Tuple
from typing import Set, Any

import cv2

from Configs.config import ANSWER_COL_WIDTH, ANSWERS_COL_X_COORDINATE, DOC_WIDTH, DOC_HEIGHT, BOUNDED_BOX_MAX_AREA, \
    DIS_BETWEEN_BOUNDED_BOXES, DIS_TO_FIRST_BOUNDED_BOX, BOUNDED_BOX_MIN_AREA


def start_process(raw_bird_eye_view_img):
    # First Grayscale it

    gray = cv2.cvtColor(raw_bird_eye_view_img, cv2.COLOR_BGR2GRAY)

    # then make it binary
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # The Rectangles on the sides
    bounded_boxes, raw_bird_eye_view_img = find_rectangles(binary, raw_bird_eye_view_img)

    height, width = binary.shape[:2]


    detected_columns: Set[Tuple[int, int, int, int]] = set()
    for rect in bounded_boxes:
        lx, ly, lw, lh = rect[0]
        rx, ry, rw, rh = rect[1]

        for col_index, x_cor in enumerate(ANSWERS_COL_X_COORDINATE):
            dis_between_bounded_boxes = (lx+lw) - rx
            dis_coef = abs(dis_between_bounded_boxes / DIS_BETWEEN_BOUNDED_BOXES)

            cv2.rectangle(raw_bird_eye_view_img, (lx, ly), (lx + lw, ly + lh), (255, 0, 255), 1)
            cv2.rectangle(raw_bird_eye_view_img, (rx, ry), (rx + rw, ry + rh), (255, 0, 255), 1)

            if not width == DOC_WIDTH:
                start = lx

                x_cor = x_cor - DIS_TO_FIRST_BOUNDED_BOX
                start_x = x_cor
                end_x = x_cor + ANSWER_COL_WIDTH

                start_x = (start_x * dis_coef)

                end_x = (end_x * dis_coef)

                start_x = start_x + start

                end_x = end_x + start
            else:
                start_x = x_cor
                end_x = x_cor + ANSWER_COL_WIDTH

            top_y = ly + (col_index / (len(ANSWERS_COL_X_COORDINATE))) * (ry - ly)
            bottom_y = (ly + lh) + (col_index / (len(ANSWERS_COL_X_COORDINATE))) * ((ry + rh) - (ly + lh))

            detected_columns.add((round(start_x), ceil(top_y), round(end_x), ceil(bottom_y)))

    answered, image = get_answered(list(detected_columns), raw_bird_eye_view_img)

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
            if (coefficient * BOUNDED_BOX_MAX_AREA) > area > (coefficient * BOUNDED_BOX_MIN_AREA):
                rectangles.append((x, y, w, h))

    rectangles = group_tuples_by_second_value(rectangles, 20)

    new_rects = []
    _,_,target_width,target_height = rectangles[0][0]
    for rec in rectangles:
        print(rec)
        print(len(rec))
        if len(rec) == 60:
            rec = resize_bounded_boxes(rec)
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

    for index, answer in enumerate(x_sorted_answers):
        x1, y1, x2, y2 = answer
        row = answers.index(answer) % 50 + 1

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]  # Value channel contains brightness info

        # Normalize the value channel to reduce shadows
        v_channel = cv2.equalizeHist(v_channel)
        hsv[:, :, 2] = v_channel

        # Convert back to BGR
        shadow_corrected = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Extract the Region of Interest (ROI)
        ROI = gray[y1:y2, x1:x2]

        # Threshold the ROI to identify filled areas
        _, thresh = cv2.threshold(ROI, 120, 255, cv2.THRESH_BINARY)

        if thresh is not None:
            total_pixels = thresh.size
            white_pixels = cv2.countNonZero(thresh)
            fill_ratio = white_pixels / total_pixels

            filled = fill_ratio < 0.6

            if col == 6 and not temp_answers:
                col = 0
            if (index % 4) == 0:
                col += 1

            question_number = ((col * 50) - 50) + row
            temp_answers.append((len(temp_answers) + 1, filled))


            # When 4 answers have been collected (for one question), store them
            if len(temp_answers) == 4:
                if question_number not in answered_questions:
                    answered_questions[question_number] = {}

                for answer_number, filled in temp_answers:
                    answered_questions[question_number][str(answer_number)] = filled

                temp_answers.clear()

            color = (0, 255, 0) if not filled else (0, 0, 255)  # Green if filled, Red otherwise
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        else:
            pass

    answered_questions = sorted(answered_questions.items(), key=lambda x: x[0])
    answered_questions = OrderedDict(answered_questions)

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


def resize_bounded_boxes(bound_boxes_to_resize):
    resized_boxes = []

    total_width = sum(box[2] for box in bound_boxes_to_resize)
    total_height = sum(box[3] for box in bound_boxes_to_resize)
    target_x = sum(box[0] for box in bound_boxes_to_resize)
    num_boxes = len(bound_boxes_to_resize)

    # Calculate averages
    avg_width = total_width / num_boxes
    avg_height = total_height / num_boxes
    avg_x = target_x / num_boxes

    for box in bound_boxes_to_resize:
        x, y, w, h = box
        x = avg_x
        y = y + ((h - avg_height) / 2)

        resized_boxes.append((round(x),round(y), round(avg_width), round(avg_height)))

    return resized_boxes

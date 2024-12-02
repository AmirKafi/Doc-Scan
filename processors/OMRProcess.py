import json
from typing import List, Tuple

import cv2
import numpy as np

from MarkerChecker import OMR
from processors.ContourCornersDetector import ContourCornersDetector


def is_bird_eye_view(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found!")

    # Detect edges
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Detect lines using Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

    if lines is None:
        print("No lines detected. Image might not be suitable for analysis.")
        return False

    # Analyze line orientation
    horizontal_count = 0
    vertical_count = 0
    total_count = 0

    for rho, theta in lines[:, 0]:
        total_count += 1
        # Horizontal lines have theta near 0 or pi
        if abs(theta) < np.pi / 6 or abs(theta - np.pi) < np.pi / 6:
            horizontal_count += 1
        # Vertical lines have theta near pi/2
        elif abs(theta - np.pi / 2) < np.pi / 6:
            vertical_count += 1

    # Check if the majority of lines are horizontal or vertical
    if total_count > 0:
        horizontal_ratio = horizontal_count / total_count
        vertical_ratio = vertical_count / total_count

        # If most lines are horizontal/vertical, it is likely a bird's-eye view
        if horizontal_ratio > 0.6 or vertical_ratio > 0.6:
            return False

    return True

def dump_result(answered_questions: List[Tuple[int, Tuple[bool, bool, bool, bool]]]):
    with open('../output/questions.json', 'w') as file:
        json.dump(answered_questions, file, indent=4)

def start_process(raw_image_path):
    image = cv2.imread(raw_image_path)
    bird_eye = is_bird_eye_view(raw_image_path)

    if not bird_eye:
        contour_detector = ContourCornersDetector()
        processed_image = contour_detector(image)
        answered_quests,image_result = OMR.start_process(processed_image)
    else:
        answered_quests,image_result = OMR.start_process(image)

    return answered_quests, image_result
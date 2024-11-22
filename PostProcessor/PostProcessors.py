import numpy as np
import cv2

def enhance(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return enhanced

def filtered(img):
    filtered = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    return filtered

def sharped(img):
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2, sigmaY=2)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return sharpened

def cleaned(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return cleaned



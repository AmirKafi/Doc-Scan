import datetime
from pathlib import Path

import cv2
import numpy as np

from MarkerChecker import OMR
from pre_processors.Denoiser import Denoiser
from processors.ContourCornersDetector import ContourCornersDetector

base_root = Path(__file__).resolve().parent.parent

start = datetime.datetime.now()

file_name = 'Scan_0002.jpg'
img = cv2.imread('../assets/'+file_name)

cv2.waitKey(0)
cv2.destroyAllWindows()


kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

# Apply the sharpening kernel
sharpened = cv2.filter2D(img, -1, kernel)
denoiser = Denoiser()
denoised = denoiser(sharpened)

answered,image_final = OMR.start_process(denoised)

# Show the resulting image with highlighted matches
cv2.namedWindow('Highlighted Rows and Matches', cv2.WINDOW_NORMAL)
cv2.imshow("Highlighted Rows and Matches", image_final)
cv2.waitKey(0)
cv2.destroyAllWindows()


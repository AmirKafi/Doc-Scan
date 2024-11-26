import datetime
from pathlib import Path

import cv2

from MarkerChecker import OMR
from MarkerChecker.OMR import dump_result

base_root = Path(__file__).resolve().parent.parent

start = datetime.datetime.now()

file_name = 'Scan_0009'
file_path = f'{base_root}/assets/{file_name}.jpg'
image = cv2.imread(file_path)

gray = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours_rec, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours_rec, -1, (0, 255, 0), 3)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

answered,image = OMR.start_process(file_path)

dump_result(answered,file_name)

# Show the resulting image with highlighted matches
cv2.namedWindow('Highlighted Rows and Matches', cv2.WINDOW_NORMAL)
cv2.imshow("Highlighted Rows and Matches", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

from processors.ContourCornersDetector import ContourCornersDetector

contour_page_extractor1 = ContourCornersDetector()

file_name = 'QalamChi.jpg'
img = cv2.imread('assets/'+file_name)
final = contour_page_extractor1('assets/'+file_name)
cv2.imshow('original', img)
cv2.imshow('final', final)
cv2.imwrite(f"output/{file_name}", final)
cv2.waitKey(0)
cv2.destroyAllWindows()
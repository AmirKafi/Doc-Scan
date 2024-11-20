import cv2

img = cv2.imread(cv2.samples.findFile('assets/photo_2024-11-20_17-16-00.jpg'))

# High-Contrast with Clahe
img_clahe = cv2.imread(cv2.samples.findFile('assets/photo_2024-11-20_17-16-00.jpg'),cv2.IMREAD_GRAYSCALE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img_clahe)

# Color-Based Segmentation
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow('HSV',hsv_img)
cv2.imshow('CLAHE',img_clahe)
cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
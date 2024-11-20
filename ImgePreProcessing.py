import cv2
import numpy as np

img = cv2.imread('photo_2024-11-19_12-31-26.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

lightened = cv2.add(img, (70, 70, 70))  # Add a scalar to each channel (B, G, R)

# Darken the image
darkened = cv2.subtract(img, (70, 70,70))  # Subtract a scalar from each channel
gray = cv2.cvtColor(darkened, cv2.COLOR_BGR2GRAY)

retVal, thresh = cv2.threshold(darkened, 60, 170, cv2.THRESH_BINARY_INV)

edges = cv2.Canny(lightened, 50, 150, apertureSize=3)
corners = cv2.goodFeaturesToTrack(edges, 1000, 0.01, 233)
corners = np.int_(corners)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x, y), 3, 255, -1)

print(corners)

# Show results
cv2.imshow("Lightened", lightened)
cv2.imshow("Darkened", darkened)
cv2.imshow('thresh', thresh)
cv2.imshow('img', img)
cv2.imshow('edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

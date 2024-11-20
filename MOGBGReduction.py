import cv2
import numpy as np

# Load image
image = cv2.imread('photo_2024-11-20_00-56-20.jpg')

# Create mask
mask = np.zeros(image.shape[:2], np.uint8)

# Define a rectangle around the foreground object
rect = (50, 50, 450, 290)

# GrabCut segmentation
cv2.grabCut(image, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)

# Modify mask to foreground and background
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Apply mask
result = image * mask2[:, :, np.newaxis]

# Show the result
cv2.imshow("Segmented", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

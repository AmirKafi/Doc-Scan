import cv2

class OtsuThresholder:
    def __init__(self, thresh1 = 0, thresh2 = 255):
        self.thresh1 = thresh1
        self.thresh2 = thresh2


    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        T_, thresholded = cv2.threshold(image, self.thresh1, self.thresh2, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded
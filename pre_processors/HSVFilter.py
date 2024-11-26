import cv2


class HSVFilter:
    def __call__(self,image):
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return hsv_img
import cv2


class EdgeDetector:

    def __call__(self,image,thresh1=0,thresh2=200,aperture_size=3):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        canny = cv2.Canny(gray, thresh1, thresh2, apertureSize=aperture_size)
        canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        return canny
import cv2


class EdgeDetector:
    def __init__(self,output_process=False):
        self.output_process = output_process

    def __call__(self,image,thresh1=0,thresh2=200,aperture_size=3):
        canny = cv2.Canny(image, thresh1, thresh2, apertureSize=aperture_size)
        canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        if self.output_process:cv2.imwrite('output/edges.jpg',canny)
        return canny
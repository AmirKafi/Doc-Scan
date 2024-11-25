import cv2


class HSVFilter:
    def __init__(self,output_process=False):
        self.output_process = output_process

    def __call__(self,image):
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if self.output_process: cv2.imwrite('output/HSVFilter.jpg',hsv_img)

        return hsv_img
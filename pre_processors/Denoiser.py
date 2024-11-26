import cv2


class Denoiser:
    def __init__(self,strength = 7):
        self.strength = strength


    def __call__(self,image):
        temp = cv2.fastNlMeansDenoising(image,h = self.strength)
        return temp

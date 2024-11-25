import cv2


class Denoiser:
    def __init__(self,strength = 7 , output_process = False):
        self.strength = strength
        self.output_process = output_process


    def __call__(self,image):
        temp = cv2.fastNlMeansDenoising(image,h = self.strength)
        if self.output_process:cv2.imwrite('output/denoised.jpg',temp)
        return temp

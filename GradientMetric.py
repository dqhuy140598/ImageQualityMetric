import cv2
import numpy as np
import math
from utils import plot_image
class GradientMetric:

        def __init__(self):
            self.lamda = 1e3
            self.gamma = 0.06
            self.N_g = math.log(self.lamda*(1-self.gamma)+1)
            self.N_c = 100
            self.grid_per_axis = int(math.sqrt(self.N_c))
            self.K_g = 2

        def get_gradient_image(self,grayscale_image):
            """
            get the gradient image from the grayscale image
            :param grayscale_image: a numpy array with format grayscale
            :return: a numpy array denotes the gradient image of a grayscale image
            """

            dX = cv2.Sobel(grayscale_image, cv2.CV_32F, 1, 0, (3, 3))
            dY = cv2.Sobel(grayscale_image, cv2.CV_32F, 0, 1, (3, 3))
            mag, direction = cv2.cartToPolar(dX, dY, angleInDegrees=True)
            mag_disp = mag / 1448  # for double values to have a result between 0-1
            plot_image(mag_disp,'Gradient Image')
            mag_disp_byte = mag / 5.6  # for uchar values between 0-255
            mag_disp_byte = mag_disp_byte /255
            return mag_disp_byte

        def get_mapped_gradient(self,gradient_image):
            """
            get the mapped gradient image form the gradient image
            :param gradient_image: a numpy array denotes the mapped gradient image
            :return: a numpy array denotes the gradient image
            """
            greater = gradient_image  >= self.gamma
            lower = gradient_image < self.gamma

            mapped_gradient_image = gradient_image.copy()
            mapped_gradient_image[greater] = (1/self.N_g) * np.log(self.lamda*(mapped_gradient_image[greater] - self.gamma) + 1)
            mapped_gradient_image[lower] = 0

            plot_image(mapped_gradient_image,'Mapped Gradient Image')

            return mapped_gradient_image

        def get_gridded_gradient_image(self,mapped_gradient_image):
            """
            get the gridded gradient image from the mapped gradient image
            :param mapped_gradient_image: a numpy array denotes the mapped gradient image
            :return: a numpy array denotes the gridded gradient image
            """
            h,w = mapped_gradient_image.shape
            gridded_gradient_image = mapped_gradient_image.copy()
            pixel_per_grid_h = int(h//self.grid_per_axis)
            pixel_per_grid_w = int(w//self.grid_per_axis)
            start_h = 0
            start_w = 0
            for i in range(self.grid_per_axis):
                end_h = (i + 1) * pixel_per_grid_h
                for j in range(self.grid_per_axis):
                    end_w = (j+1)*pixel_per_grid_w
                    gridded_gradient_image [start_h:end_h,start_w:end_w] = np.sum(gridded_gradient_image[start_h:end_h,start_w:end_w])
                    start_w = end_w
                start_w = 0
                start_h = end_h

            plot_image(gridded_gradient_image,'Gridded Gradient Image')
            return gridded_gradient_image

        def estimate_gradient_score(self,image):
            """
            estimate gradient-based metric of the input image
            :param image: the numpy array with format BGR
            :return: the score of the gradient-based metric
            """

            grayscale = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            gradient_image = self.get_gradient_image(grayscale)
            mapped_gradient_image = self.get_mapped_gradient(gradient_image)
            gridd_gradient_image  = self.get_gridded_gradient_image(mapped_gradient_image)

            gradient_score = self.K_g * np.mean(gridd_gradient_image) / np.std(gridd_gradient_image)

            return gradient_score







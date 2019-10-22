import cv2
import numpy as np
from utils import plot_image
from scipy.signal import convolve2d
import math
class NoiseMetric:

    def __init__(self):
        self.kernel_M = np.array([[1,-2,1],[-2,4,-2],[1,-2,1]])
        self.p = 0.1
        self.t_l = 15
        self.t_h = 235

    def get_gradient_image(self,grayscale_image):
        """
        Get the gradient magnitude image from gray scale image
        :param grayscale_image: the numpy array with format grayscale
        :return: the gradient image of the input image
        """
        dX = cv2.Sobel(grayscale_image, cv2.CV_32F, 1, 0, (3, 3))
        dY = cv2.Sobel(grayscale_image, cv2.CV_32F, 0, 1, (3, 3))
        mag, direction = cv2.cartToPolar(dX, dY, angleInDegrees=True)
        mag_disp = mag / 1448  # for double values to have a result between 0-1
        plot_image(mag_disp, 'Gradient Image')
        mag_disp_byte = mag / 5.6  # for uchar values between 0-255
        mag_disp_byte = mag_disp_byte / 255
        return mag_disp_byte

    def get_inital_image_noise(self,channel):
        """
        get the inital image by convolve the channel with the kernel M
        :param channel: one of channels of the RGB color image
        :return: the inital image
        """
        inital_image_noise = convolve2d(channel,self.kernel_M,mode='same',boundary='fill',fillvalue=0)
        plot_image(inital_image_noise,'Inital Image Noise')
        return inital_image_noise


    def get_homogeneous_region_mask(self,channel):
        """
        get the homogeneous region mask from the channel
        :param channel: one of channels of the RGB color image
        :return: the homogeneous mask
        """
        gradient_image = self.get_gradient_image(channel)
        gradient_vector = gradient_image.ravel()
        gradient_vector = np.sort(gradient_vector)
        ordinal_rank = math.ceil((0.1/100)*len(gradient_vector))
        delta = gradient_vector[ordinal_rank]
        homogeneous_region_mask = gradient_image.copy()
        lower = gradient_image <= delta
        greater = gradient_image > delta
        homogeneous_region_mask[lower] = 1
        homogeneous_region_mask[greater] = 0
        plot_image(homogeneous_region_mask,'Homogeneous Region Mask')
        return homogeneous_region_mask

    def get_unsaturated_region_mask(self,channel):
        """
        get the unsaturated region mask from the channel
        :param channel: one of channels of the RGB color image
        :return: the homogeneous mask
        """
        between = np.logical_and(channel >=self.t_l,channel <=self.t_h)
        out = np.logical_or(channel<self.t_l,channel>self.t_h)
        unsaturated_region_mask = channel.copy()
        unsaturated_region_mask[between] = 1
        unsaturated_region_mask[out] = 0
        plot_image(unsaturated_region_mask,'Unsaturated Region Mask')
        return unsaturated_region_mask

    def estimate_noise_channel(self,channel):
        """
        estimate noise variance in a channel
        :param channel: one of channels of the RGB color image
        :return: a float denotes the noise variance of the channel
        """
        h,w = channel.shape
        homogeneous_region_mask = self.get_homogeneous_region_mask(channel)
        unsaturated_region_mask = self.get_unsaturated_region_mask(channel)
        inital_noise_image = self.get_inital_image_noise(channel)
        N_s = (h-2)*(w-2)
        noise_score = math.sqrt(math.pi / 2) *(1/ N_s) * np.sum(homogeneous_region_mask * unsaturated_region_mask * np.abs(inital_noise_image))
        return noise_score

    def estimate_noise_score(self,image):
        """
        estimate noise variance in a image
        :param image: the numpy array with BGR format
        :return: a float denotes the noise variance of the image
        """
        b_score = self.estimate_noise_channel(image[...,0])
        g_score = self.estimate_noise_channel(image[...,1])
        r_score = self.estimate_noise_channel(image[...,2])
        average_noise_score = (b_score + g_score + r_score)/3
        return average_noise_score







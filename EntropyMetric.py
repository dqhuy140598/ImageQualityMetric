import cv2
import numpy as np
from utils import plot_image
import numpy as np
import skimage
from sklearn.metrics.cluster import entropy

class EntropyMetric:

    def __init__(self):
        self.K_e = 0.125

    def estimate_entropy_score(self,image):
        grayscale = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        entropy_score = self.K_e * entropy(grayscale)
        return entropy_score

    def estimate_entropy_score_2(self,image):
        """
        estimate the entropy of the image
        :param image: a numpy array with format BGR
        :return: a float denotes the entropy of a image
        """
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        entropy = self.K_e * skimage.measure.shannon_entropy(grayscale)
        return entropy


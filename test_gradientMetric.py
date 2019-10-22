from unittest import TestCase
import cv2
import numpy as np
from GradientMetric import GradientMetric
class TestGradientMetric(TestCase):

    def test_get_gradient_image(self):
        image_path = 'data/fusion.png'
        gray_scale = cv2.imread(image_path,0)
        gradient_metric = GradientMetric()
        gradient_image = gradient_metric.get_gradient_image(gray_scale)
        self.assertEqual(gradient_image.shape,gray_scale.shape)
        max = np.max(gradient_image)
        min = np.min(gradient_image)
        self.assertGreaterEqual(min,0.0)
        self.assertLessEqual(max,1.0)

    def test_get_mapped_gradient(self):
        image_path = 'data/fusion.png'
        gray_scale = cv2.imread(image_path,0)
        gradient_metric = GradientMetric()
        gradient_image = gradient_metric.get_gradient_image(gray_scale)
        mapped_gradient_image = gradient_metric.get_mapped_gradient(gradient_image)
        self.assertEqual(mapped_gradient_image.shape,gradient_image.shape)
        self.assertIsNotNone(mapped_gradient_image)

    def test_get_gridded_gradient_image(self):
        image_path = 'data/fusion.png'
        gray_scale = cv2.imread(image_path, 0)
        gradient_metric = GradientMetric()
        gradient_image = gradient_metric.get_gradient_image(gray_scale)
        mapped_gradient_image = gradient_metric.get_mapped_gradient(gradient_image)
        gridded_gradient_image = gradient_metric.get_gridded_gradient_image(mapped_gradient_image)
        self.assertEqual(gridded_gradient_image.shape,mapped_gradient_image.shape)

    def test_estimate_gradient_score(self):
        image_path = 'data/under.jpg'
        image = cv2.imread(image_path)
        gradient_metric = GradientMetric()
        gradient_score = gradient_metric.estimate_gradient_score(image)
        print(gradient_score)
        self.assertEqual(isinstance(gradient_score,float),True)



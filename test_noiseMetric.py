from unittest import TestCase
from NoiseMetric import NoiseMetric
import cv2
class TestNoiseMetric(TestCase):
    def test_get_inital_image_noise(self):
        image_path = 'data/fusion.png'
        image = cv2.imread(image_path)
        blue_channel = image[...,0]
        noise_metric = NoiseMetric()
        inital_image_noise = noise_metric.get_inital_image_noise(blue_channel)
        self.assertEqual(inital_image_noise.shape,blue_channel.shape)

    def test_get_unsaturated_region_mask(self):
        image_path = 'data/fusion.png'
        image = cv2.imread(image_path)
        blue_channel = image[..., 0]
        noise_metric = NoiseMetric()
        unsaturated_region_mask = noise_metric.get_unsaturated_region_mask(blue_channel)
        self.assertEqual(unsaturated_region_mask.shape,blue_channel.shape)

    def test_get_homogenenous_region_mask(self):
        image_path = 'data/fusion.png'
        image = cv2.imread(image_path)
        blue_channel = image[..., 0]
        noise_metric = NoiseMetric()
        unsaturated_region_mask = noise_metric.get_homogeneous_region_mask(blue_channel)
        self.assertEqual(unsaturated_region_mask.shape, blue_channel.shape)

    def test_estimate_noise_channel(self):
        image_path = 'data/fusion.png'
        image = cv2.imread(image_path)
        blue_channel = image[..., 0]
        noise_metric = NoiseMetric()
        noise_score = noise_metric.estimate_noise_channel(blue_channel)
        print(noise_score)
        self.assertEqual(isinstance(noise_score,float),True)

    def test_estimate_noise_score(self):
        image_path = 'data/under.jpg'
        image = cv2.imread(image_path)
        noise_metric = NoiseMetric()
        noise_score = noise_metric.estimate_noise_score(image)
        print(noise_score)
        self.assertEqual(isinstance(noise_score, float), True)

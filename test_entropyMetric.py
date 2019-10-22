from unittest import TestCase
from EntropyMetric import  EntropyMetric
import  cv2
class TestEntropyMetric(TestCase):
    def test_estimate_entropy_score(self):
        image_path = 'data/fusion.png'
        image = cv2.imread(image_path)
        entropy_metric = EntropyMetric()
        entropy_score = entropy_metric.estimate_entropy_score(image)
        print(entropy_score)
        self.assertEqual(isinstance(entropy_score,float),True)
        self.assertGreaterEqual(entropy_score,0.0)

    def test_estimate_entropy_score_2(self):
        image_path = 'data/fusion.png'
        image = cv2.imread(image_path)
        entropy_metric = EntropyMetric()
        entropy_score = entropy_metric.estimate_entropy_score_2(image)
        print(entropy_score)
        self.assertEqual(isinstance(entropy_score, float), True)
        self.assertGreaterEqual(entropy_score, 0.0)
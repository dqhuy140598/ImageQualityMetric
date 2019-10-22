from GradientMetric import GradientMetric
from EntropyMetric import EntropyMetric
from NoiseMetric import NoiseMetric
import cv2
class ImageQuatityMetric:

        def __init__(self):
            self.alpha = 0.4
            self.beta = 0.4
            self.gradient_metric = GradientMetric()
            self.entropy_metric = EntropyMetric()
            self.noise_metric = NoiseMetric()

        def estimate_image_quality(self,image):
            """
            estimate image quality of the input image
            :param image: the numpy array with format BGR
            :return: a float denotes the image quality of the input image
            """
            gradient_score = self.gradient_metric.estimate_gradient_score(image)
            entropy_score = self.entropy_metric.estimate_entropy_score_2(image)
            noise_score = self.noise_metric.estimate_noise_score(image)

            image_quality_score = self.alpha * gradient_score + (1-self.alpha) * entropy_score - self.beta * noise_score

            return image_quality_score


if __name__ == '__main__':

    under_path = 'data/under.jpg'
    fusion_path = 'data/fusion.png'
    over_path = 'data/over.jpg'

    under_image = cv2.imread(under_path)
    fusion_image = cv2.imread(fusion_path)
    over_image = cv2.imread(over_path)

    image_quality_metric = ImageQuatityMetric()

    under_score = image_quality_metric.estimate_image_quality(under_image)
    fusion_score = image_quality_metric.estimate_image_quality(fusion_image)
    over_score = image_quality_metric.estimate_image_quality(over_image)

    print('under_score: ',under_score)
    print('fusion_score: ',fusion_score)
    print('over_score: ',over_score)
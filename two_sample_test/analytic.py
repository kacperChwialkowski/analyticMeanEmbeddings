from two_sample_test.utils import mahalanobis_distance, smooth_difference

__author__ = 'kcx'
import numpy


class MeanEmbeddingTest:

    def __init__(self, data_x, data_y, scale=1, number_of_random_frequencies=5):
        self.data_x = scale*data_x
        self.data_y = scale*data_y
        self.number_of_frequencies = number_of_random_frequencies
        self.scale = scale

    def get_estimate(self, data, point):
        z = data - self.scale * point
        z2 = numpy.linalg.norm(z, axis=1)**2
        return numpy.exp(-z2/2.0)


    def get_difference(self, point):
        return self.get_estimate(self.data_x, point) - self.get_estimate(self.data_y, point)


    def vector_of_differences(self, dim):
        points = numpy.random.randn(self.number_of_frequencies, dim)
        a = [self.get_difference(point) for point in points]
        return numpy.array(a).T

    def compute_pvalue(self):

        _, dimension = numpy.shape(self.data_x)
        obs = self.vector_of_differences(dimension)

        return mahalanobis_distance(obs, self.number_of_frequencies)

class SmoothCFTest:

    def _gen_random(self, dimension):
        return numpy.random.randn(dimension, self.num_random_features)


    def __init__(self, data_x, data_y, scale=2.0, num_random_features=5, frequency_generator=None):
        self.data_x = scale*data_x
        self.data_y = scale*data_y
        self.num_random_features = num_random_features

        _, dimension_x = numpy.shape(self.data_x)
        _, dimension_y = numpy.shape(self.data_y)
        assert dimension_x == dimension_y
        self.random_frequencies = self._gen_random(dimension_x)


    def compute_pvalue(self):

        difference = smooth_difference(self.random_frequencies, self.data_x, self.data_y)
        return mahalanobis_distance(difference, 2 * self.num_random_features)
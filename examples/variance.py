__author__ = 'kcx'

from two_sample_test.analytic import SmoothCFTest, MeanEmbeddingTest
from multiprocessing import Pool
import time
import gc
import sys
import numpy
from numpy import save, mean


POOL_SIZE = 4
REPETITIONS = 100
NUM_RANDOM_FREQUENCIES = 3
DIMENSION = range(50, 1101, 100)
SAMPLE_SIZE = 10000



def reject_and_time(test):
    t1 = time.time()
    pvalue = test.compute_pvalue()
    t2 = time.time()
    return [pvalue , (t2 - t1)]

def generate_X(dim):
    return numpy.random.randn(SAMPLE_SIZE, dim)

def generate_Y(dim):
    X = numpy.random.randn(SAMPLE_SIZE, dim)
    X[:, 42] *= 2
    return X

def simulation(dimensions):
    rejection_rates = [reject_and_time(test) for test in [
        SmoothCFTest(generate_X(dimensions),generate_Y(dimensions), scale=2.0 ** (-8), num_random_features=NUM_RANDOM_FREQUENCIES),
        MeanEmbeddingTest(generate_X(dimensions),generate_Y(dimensions), scale=2.0 ** (-8), number_of_random_frequencies=NUM_RANDOM_FREQUENCIES)
    ]]
    gc.collect()
    return rejection_rates


def simulations(dim):
    samples = mean([simulation(dim) for _ in range(REPETITIONS)], 0)
    print 'variance: dimension ', str(dim), ' with \n ', samples
    sys.stdout.flush()
    return samples


def run_H1_simulation(x):
    return [simulations(dim) for dim in DIMENSION]


if __name__ == "__main__":
    t1 = time.time()
    p = Pool(POOL_SIZE)
    p_values_and_time= p.map(run_H1_simulation, [1,2,3,4])
    p_values_and_time = numpy.mean(p_values_and_time, axis=0)
    t2 = time.time()
    print 'total time', (t2 - t1)
    print p_values_and_time
    save('./results/variance', p_values_and_time)
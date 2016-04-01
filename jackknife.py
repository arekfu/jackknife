import numpy as np

def subsamples(sample):
    subsamples = []
    for i in range(len(sample)):
        subsample = [ x for j, x in enumerate(sample) if i!=j ]
        yield subsample

def subsample_mean(estimator, sample):
    ests = [ estimator(sub) for sub in subsamples(sample) ]
    return np.mean(ests)

def bias(estimator, sample):
    n = len(sample)
    bias_est = (n-1)*(subsample_mean(estimator, sample) - estimator(sample))
    return bias_est

def estimate(estimator, sample):
    """Returns the jackknife estimate of the mean

    This is equal to the mean minus the jackknife estimate of the bias.
    """

    n = len(sample)
    mean_est = n*estimator(sample) - (n-1)*subsample_mean(estimator, sample)
    return mean_est

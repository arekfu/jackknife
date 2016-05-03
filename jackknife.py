import numpy as np

def subsamples(sample):
    subsamples = []
    for i in range(len(sample)):
        subsample = [ x for j, x in enumerate(sample) if i!=j ]
        yield subsample

def subsample_means(estimator, sample):
    return np.array([ estimator(sub) for sub in subsamples(sample) ])

def subsample_mean(estimator, sample):
    return np.mean(subsample_means(estimator, sample))

def bias(estimator, sample):
    n = len(sample)
    bias_est = (n-1)*(subsample_mean(estimator, sample) - estimator(sample))
    return bias_est

def pseudovalues(estimator, sample):
    """Returns the jackknife pseudovalues"""

    n = len(sample)
    ss_means = subsample_means(estimator, sample)
    pseudos = n*estimator(sample) - (n-1)*ss_means
    return pseudos

def estimate(estimator, sample, with_subsample_mean=False):
    """Returns the jackknife estimate of the mean

    This is equal to the mean minus the jackknife estimate of the bias.
    """

    n = len(sample)
    ss_mean = subsample_mean(estimator, sample)
    mean_est = n*estimator(sample) - (n-1)*ss_mean
    if with_subsample_mean:
        return mean_est, ss_mean
    else:
        return mean_est

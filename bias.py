#!/usr/bin/env python3

import numpy as np
import sys

import jackknife

n_sample = 5
n_replicas = 10000

ests = []
jack_ests = []
jack_ests_sq = []

jack_pseudo_means = []
jack_pseudo_vars = []

est_fun = 'max' if len(sys.argv) == 1 else sys.argv[1]

if est_fun == 'max':
    estimator = np.max
    sampler = lambda n : np.random.uniform(size=n)
    expected = 1.
    bias = 1. - 1./n_sample
elif est_fun == 'exp':
    x0 = 0.
    sigma = 1.
    lmbda = 1.
    def estimator(sample):
        return np.exp(lmbda * np.mean(sample))
    sampler = lambda n: np.random.normal(x0, sigma, size=n)
    expected = estimator(lmbda * x0)
    bias = expected*(estimator(0.5*lmbda*sigma**2/n_sample) - 1.)
    expected_ss_mean = np.exp(x0*lmbda + 0.5*sigma**2*lmbda**2/(n_sample-1))
    expected_ss_mean_sq = np.exp(2*x0*lmbda + 2*sigma**2*lmbda**2/(n_sample-1))/n_sample \
            + np.exp(2*x0*lmbda+sigma**2*lmbda**2*(2*n_sample-3)/(n_sample-1)**2)*(n_sample-1)/n_sample
else:
    print('unrecognized estimator')
    sys.exit(1)

for irep in range(n_replicas):
    sample = sampler(n_sample)

    est = estimator(sample)
    ests.append(est)

    jack_est = jackknife.estimate(estimator, sample)
    jack_ests.append(jack_est)
    jack_ests_sq.append(jack_est**2)

    pseudo = jackknife.pseudovalues(estimator, sample)
    jack_pseudo_mean = np.mean(pseudo)
    jack_pseudo_var = np.var(pseudo, ddof=1)
    jack_pseudo_means.append(jack_pseudo_mean)
    jack_pseudo_vars.append(jack_pseudo_var)

av_est = np.mean(ests)
std_est = np.sqrt(np.var(ests, ddof=1) / n_replicas)
sigmas_est = np.abs(av_est - expected) / std_est

av_jack_est = np.mean(jack_ests)
std_jack_est = np.sqrt(np.var(jack_ests, ddof=1) / n_replicas)
sigmas_jack_est = np.abs(av_jack_est - expected) / std_jack_est

av_jack_est_sq = np.mean(jack_ests_sq)

print(u'expected value              = {:.4f}'.format(expected))
print(u'average estimator           = {:.4f} ± {:.4f} ({:.1f} sigma)'.format(av_est, std_est, sigmas_est))
print(u'average jackknife estimator = {:.4f} ± {:.4f} ({:.1f} sigma)'.format(av_jack_est, std_jack_est, sigmas_jack_est))
print(u'average jackknife squared   = {:.4f}'.format(av_jack_est_sq))

print()
print(u'average pseudovalue average = {:.4f}'.format(np.mean(jack_pseudo_means)))
print(u'sqrt average pseudovalue var= {:.4f}'.format(np.sqrt(np.mean(jack_pseudo_vars))))

print()
print(u'expected bias               = {:.4e}'.format(bias))
print(u'average estimator bias      = {:.4e}'.format(av_est - expected))
print(u'average jackknife bias      = {:.4e}'.format(av_jack_est - expected))

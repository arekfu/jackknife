#!/usr/bin/env python3

import numpy as np
import sys

import jackknife

n_sample = 100
n_replicas = 1000

ests = []
jack_ests = []

est_fun = 'max' if len(sys.argv) == 1 else sys.argv[1]

if est_fun == 'max':
    estimator = np.max
    sampler = np.random.uniform
    expected = 1
elif est_fun == 'exp':
    def estimator(sample):
        return np.exp(np.mean(sample))
    sampler = np.random.normal
    expected = 1
else:
    print('unrecognized estimator')
    sys.exit(1)

for irep in range(n_replicas):
    sample = sampler(size=n_sample)

    est = estimator(sample)
    ests.append(est)

    jack_est = jackknife.estimate(estimator, sample)
    jack_ests.append(jack_est)

av_est = np.mean(ests)
av_jack_est = np.mean(jack_ests)

print('expected value              = {}'.format(expected))
print('average estimator           = {}'.format(av_est))
print('average jackknife estimator = {}'.format(av_jack_est))
print('average estimator bias      = {:e}'.format(av_est - expected))
print('average jackknife bias      = {:e}'.format(av_jack_est - expected))

#!/usr/bin/env python3

import numpy as np
import jackknife

n_sample = 30
n_replicas = 1000

max_ests = []
jack_max_ests = []

for irep in range(n_replicas):
    sample = np.random.uniform(size=n_sample)

    max_est = np.max(sample)
    max_ests.append(max_est)

    jack_max_est = jackknife.estimate(np.max, sample)
    jack_max_ests.append(jack_max_est)

print('average max estimator: {}'.format(np.mean(max_ests)))
print('average jackknife max estimator: {}'.format(np.mean(jack_max_ests)))

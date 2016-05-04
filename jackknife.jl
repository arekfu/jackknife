function subsamples(sample::Vector)
    subsamples = similar(sample, typeof(sample))
    for i in 1:length(sample)
      subsamples[i] = [collect(take(sample, i-1)); collect(drop(sample, i))]
    end
    subsamples
end

subsample_means(estimator, sample) = map(estimator, subsamples(sample))

subsample_mean(estimator, sample) = mean(subsample_means(estimator, sample))

function bias(estimator, sample)
  n = length(sample)
  (n-1)*(subsample_mean(estimator, sample) - estimator(sample))
end

function pseudovalues(estimator, sample)
  """Returns the jackknife pseudovalues"""

  n = length(sample)
  ss_means = subsample_means(estimator, sample)
  n*estimator(sample) - (n-1)*ss_means
end

function estimate(estimator, sample; with_subsample_mean=false)
  """Returns the jackknife estimate of the mean

  This is equal to the mean minus the jackknife estimate of the bias.
  """

  n = length(sample)
  ss_mean = subsample_mean(estimator, sample)
  mean_est = n*estimator(sample) - (n-1)*ss_mean
  if with_subsample_mean
    return mean_est, ss_mean
  else
    return mean_est
  end
end

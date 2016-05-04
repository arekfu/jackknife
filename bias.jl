#!/usr/bin/env julia

include("jackknife.jl")
import Distributions
using Formatting

n_sample = 5
n_replicas = 10000

ests = Vector{Float64}()
jack_ests = Vector{Float64}()
jack_ests_sq = Vector{Float64}()

jack_pseudo_means = Vector{Float64}()
jack_pseudo_vars = Vector{Float64}()

μ = 0.
σ = 1.
λ = 1.
estimator(sample) = exp(λ * mean(sample))
normal = Distributions.Normal(μ, σ)
sampler = n -> rand(normal, n)
expected = estimator(λ * μ)
expected_bias = expected*(estimator(0.5*λ*σ^2/n_sample) - 1.)
expected_ss_mean = exp(μ*λ + 0.5*σ^2*λ^2/(n_sample-1))
expected_ss_mean_sq = exp(2*μ*λ + 2*σ^2*λ^2/(n_sample-1))/n_sample \
        + exp(2*μ*λ+σ^2*λ^2*(2*n_sample-3)/(n_sample-1)^2)*(n_sample-1)/n_sample

for irep in 1:n_replicas
  sample = sampler(n_sample)

  est = estimator(sample)
  push!(ests, est)

  jack_est = estimate(estimator, sample)
  push!(jack_ests, jack_est)
  push!(jack_ests_sq, jack_est^2)

  pseudo = pseudovalues(estimator, sample)
  jack_pseudo_mean = mean(pseudo)
  jack_pseudo_var = var(pseudo)
  push!(jack_pseudo_means, jack_pseudo_mean)
  push!(jack_pseudo_vars, jack_pseudo_var)
end

av_est = mean(ests)
std_est = sqrt(var(ests) / n_replicas)
sigmas_est = abs(av_est - expected) / std_est

av_jack_est = mean(jack_ests)
std_jack_est = sqrt(var(jack_ests) / n_replicas)
sigmas_jack_est = abs(av_jack_est - expected) / std_jack_est

av_jack_est_sq = mean(jack_ests_sq)

printfmtln("expected value              = {:.4f}", expected)
printfmtln("average estimator           = {:.4f} ± {:.4f} ({:.1f} sigma)", av_est, std_est, sigmas_est)
printfmtln("average jackknife estimator = {:.4f} ± {:.4f} ({:.1f} sigma)", av_jack_est, std_jack_est, sigmas_jack_est)
printfmtln("average jackknife squared   = {:.4f}", av_jack_est_sq)

println()
printfmtln("average pseudovalue average = {:.4f}", mean(jack_pseudo_means))
printfmtln("sqrt average pseudovalue var= {:.4f}", sqrt(mean(jack_pseudo_vars)))

println()
printfmtln("expected bias               = {:.4e}", expected_bias)
printfmtln("average estimator bias      = {:.4e}", av_est - expected)
printfmtln("average jackknife bias      = {:.4e}", av_jack_est - expected)

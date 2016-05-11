#!/usr/bin/env julia

include("jackknife.jl")
include("packets.jl")
import Distributions
using Formatting

n_sample = 3
n_replicas = 10000
n_packets = 10
packet_size = max(1, n_sample ÷ n_packets)

ests = Vector{Float64}()

packetwises = Vector{Float64}()

jack_ests = Vector{Float64}()
jack_ests_sq = Vector{Float64}()

jack_pseudo_means = Vector{Float64}()
jack_pseudo_vars = Vector{Float64}()

μ = 0.
σ = 1.
λ = 1.
ɛ = λ^2 * σ^2 / n_sample

estimator(sample) = exp(λ * mean(sample))
normal = Distributions.Normal(μ, σ)
sampler = n -> rand(normal, n)
expected = estimator(λ * μ)
expected_bias = expected*(estimator(0.5*λ*σ^2/n_sample) - 1.)

expected_var_jack_est = expected^2 * (
  (n_sample^3 * estimator(2*λ*σ^2/n_sample)
    - 2*n_sample^2*(n_sample-1)*estimator(0.5*λ*σ^2*(4*n_sample-3)/n_sample/(n_sample-1))
    + (n_sample-1)^3 * estimator(λ*σ^2*(2*n_sample-3)/(n_sample-1)^2)
    + (n_sample-1)^2 * estimator(2*λ*σ^2/(n_sample-1))
    )/n_sample - (
    (n_sample-1) * estimator(0.5*λ*σ^2/(n_sample-1))
    - n_sample * estimator(0.5*λ*σ^2/n_sample)
    )^2
    )

expected_packetwise_bias = expected*(estimator(0.5*λ*σ^2*packet_size/n_sample) - 1.)
expected_ss_mean = exp(μ*λ + 0.5*σ^2*λ^2/(n_sample-1))
expected_ss_mean_sq = exp(2*μ*λ + 2*σ^2*λ^2/(n_sample-1))/n_sample \
        + exp(2*μ*λ+σ^2*λ^2*(2*n_sample-3)/(n_sample-1)^2)*(n_sample-1)/n_sample

for irep in 1:n_replicas
  sample = sampler(n_sample)

  est = estimator(sample)
  push!(ests, est)

  packetwise = packetwise_mean(packet_size, estimator, sample)
  push!(packetwises, packetwise)

  jack_est = estimate(estimator, sample)
  push!(jack_ests, jack_est)
  push!(jack_ests_sq, jack_est^2)

  pseudo = pseudovalues(estimator, sample)
  jack_pseudo_mean = mean(pseudo)
  jack_pseudo_var = var(pseudo)
  push!(jack_pseudo_means, jack_pseudo_mean)
  push!(jack_pseudo_vars, jack_pseudo_var/n_sample)
end

av_est = mean(ests)
std_est = sqrt(var(ests))
sigmas_est = abs(av_est - expected) / std_est

av_packetwise = mean(packetwises)

av_jack_est = mean(jack_ests)
std_jack_est = sqrt(var(jack_ests))
sigmas_jack_est = abs(av_jack_est - expected) / std_jack_est

av_jack_est_sq = mean(jack_ests_sq)

av_jack_pseudo = mean(jack_pseudo_means)
std_jack_pseudo = sqrt(mean(jack_pseudo_vars))

printfmtln("ɛ                           = {:.4f}", ɛ)

println()
printfmtln("expected value              = {:.4f}", expected)
printfmtln("average estimator           = {:.4f} ± {:.4f} ({:.1f} sigma)", av_est, std_est, sigmas_est)
printfmtln("variance average estimator  = {:.4f}", std_est^2)

println()
printfmtln("average jackknife estimator = {:.4f} ± {:.4f} ({:.1f} sigma)", av_jack_est, std_jack_est, sigmas_jack_est)
printfmtln("variance jackknife estimator= {:.4f}", std_jack_est^2)
printfmtln("expected variance jackknife = {:.4f}", expected_var_jack_est)
printfmtln("average jackknife squared   = {:.4f}", av_jack_est_sq)

println()
printfmtln("average packetwise mean     = {:.4f}", av_packetwise)
printfmtln("average packetwise bias     = {:.4f}", av_packetwise - expected)
printfmtln("expected packetwise bias    = {:.4e}", expected_packetwise_bias)

println()
printfmtln("average jackknife pseudo    = {:.4f}", av_jack_pseudo)
printfmtln("variance jackknife pseudo   = {:.4f}", std_jack_pseudo^2)

println()
printfmtln("expected bias for normal est= {:.4e}", expected_bias)
printfmtln("average normal est bias     = {:.4e}", av_est - expected)
printfmtln("average jackknife bias      = {:.4e}", av_jack_est - expected)

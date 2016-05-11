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

trans = identity; transinv = identity
estimator(sample) = exp(λ * mean(sample))
transestimator(sample) = trans(estimator(sample))
normal = Distributions.Normal(μ, σ)
sampler = n -> rand(normal, n)
unbiased = estimator(λ * μ)
theoretical_bias = unbiased*(estimator(0.5*λ*σ^2/n_sample) - 1.)

theoretical_var_jack_est = unbiased^2 * (
  (n_sample^3 * estimator(2*λ*σ^2/n_sample)
    - 2*n_sample^2*(n_sample-1)*estimator(0.5*λ*σ^2*(4*n_sample-3)/n_sample/(n_sample-1))
    + (n_sample-1)^3 * estimator(λ*σ^2*(2*n_sample-3)/(n_sample-1)^2)
    + (n_sample-1)^2 * estimator(2*λ*σ^2/(n_sample-1))
    )/n_sample - (
    (n_sample-1) * estimator(0.5*λ*σ^2/(n_sample-1))
    - n_sample * estimator(0.5*λ*σ^2/n_sample)
    )^2
    )

theoretical_packetwise_bias = unbiased*(estimator(0.5*λ*σ^2*packet_size/n_sample) - 1.)
theoretical_ss_mean = exp(μ*λ + 0.5*σ^2*λ^2/(n_sample-1))
theoretical_ss_mean_sq = exp(2*μ*λ + 2*σ^2*λ^2/(n_sample-1))/n_sample \
        + exp(2*μ*λ+σ^2*λ^2*(2*n_sample-3)/(n_sample-1)^2)*(n_sample-1)/n_sample

for irep in 1:n_replicas
  sample = sampler(n_sample)

  est = estimator(sample)
  push!(ests, est)

  packetwise = packetwise_mean(packet_size, estimator, sample)
  push!(packetwises, packetwise)

  jack_est = transinv(estimate(transestimator, sample))
  push!(jack_ests, jack_est)
  push!(jack_ests_sq, jack_est^2)

  pseudo = transinv(pseudovalues(transestimator, sample))
  jack_pseudo_mean = mean(pseudo)
  jack_pseudo_var = var(pseudo)
  push!(jack_pseudo_means, jack_pseudo_mean)
  push!(jack_pseudo_vars, jack_pseudo_var/n_sample)
end

av_est = mean(ests)
std_est = sqrt(var(ests))
std_av_est = std_est / sqrt(n_replicas)
sigmas_est = abs(av_est - unbiased) / std_av_est

av_packetwise = mean(packetwises)

av_jack_est = mean(jack_ests)
std_jack_est = sqrt(var(jack_ests))
std_av_jack_est = std_jack_est / sqrt(n_replicas)
sigmas_jack_est = abs(av_jack_est - unbiased) / std_av_jack_est

av_jack_est_sq = mean(jack_ests_sq)

av_jack_pseudo = mean(jack_pseudo_means)
std_jack_pseudo = sqrt(mean(jack_pseudo_vars))

printfmtln("                             ɛ = {:.4f}", ɛ)

println()
printfmtln("unbiased value                 = {:.4f}", unbiased)

println("\nNAIVE ESTIMATORS")
printfmtln("E[average estimator]           = {:.4f} ± {:.4f} ({:.1f} sigma)", av_est, std_av_est, sigmas_est)
printfmtln("E[variance average estimator]  = {:.4e}", std_est^2)

println("\nJACKKNIFE ESTIMATORS")
printfmtln("E[average jackknife]           = {:.4f} ± {:.4f} ({:.1f} sigma)", av_jack_est, std_av_jack_est, sigmas_jack_est)
#printfmtln("average jackknife squared      = {:.4f}", av_jack_est_sq)
printfmtln("E[variance jackknife]          = {:.4e}", std_jack_est^2)
printfmtln("theoretical variance jackknife = {:.4e}", theoretical_var_jack_est)

println("\nPACKETWISE ESTIMATORS")
printfmtln("number of packets              = {}", n_packets)
printfmtln("E[average packetwise]          = {:.4f}", av_packetwise)
printfmtln("E[average packetwise bias]     = {:.4f}", av_packetwise - unbiased)
printfmtln("theoretical packetwise bias    = {:.4e}", theoretical_packetwise_bias)

println("\nJACKKNIFE PSEUDOVALUE ESTIMATORS")
printfmtln("E[average jackknife pseudo]    = {:.4f}", av_jack_pseudo)
printfmtln("E[variance jackknife pseudo]   = {:.4e}", std_jack_pseudo^2)

println("\nBIASES")
printfmtln("theoretical bias for naive est = {:.4e}", theoretical_bias)
printfmtln("E[naive est bias]              = {:.4e}", av_est - unbiased)
printfmtln("E[jackknife bias]              = {:.4e}", av_jack_est - unbiased)

println()
printfmtln("BIAS DIAGNOSTICS")
bias_estimate = av_jack_est - av_est
printfmtln("average - jackknife         = {:.4f}", bias_estimate)
printfmtln("jackknife standard error    = {:.4f}", std_av_jack_est)
diagnostics = abs(bias_estimate/std_jack_est)
printfmtln("bias estimate/standard error= {:.4e}", diagnostics)
if diagnostics < 1.
  println(" ... bias is smaller than standard error ...")
else
  println(" !!! bias is larger than standard error !!!")
end

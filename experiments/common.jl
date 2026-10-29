## Shared helpers for the inference experiments (include from scripts).
using PhyloPOMP
using PhyloPOMP: ExactSEIR, SimSEIR, FastSEIR
using PhyloPOMP.FastSEIR.Demes: Expos, Infec
using Random, Statistics, Printf

const RESULTS = joinpath(@__DIR__, "results")

## model parameters used to simulate the package's example trees
const DEFAULTS = (β=4.0, σ=1.0, γ=1.0, ω=1.0, ψ=0.02, χ=0.0,
                  pop=100, S0=0.9, E0=0.0, I0=0.02, R0=0.08)

## the ad hoc guide used in the package's tests and docs
default_guide() = fsmarkov(Expos=>0.1, Infec=>1, (Expos,Infec)=>1)

## endemic equilibrium of the deterministic SEIRS model
equilibrium(; β, σ, γ, ω, pop, _...) = begin
    S = pop*γ/β
    I = (pop-S)/(1 + γ/σ + γ/ω)
    (; S, E = γ*I/σ, I, R = γ*I/ω)
end

## Guide matched to the backward-in-time lineage dynamics at equilibrium:
## a lineage in E moves to I at the per-capita incidence βSI/(NE) = σ, and
## from I to E at σE/I.  With fsmarkov's parameterization the rates are
## E→I = π_I c and I→E = π_E c, so π ∝ (E*, I*) and c = σ (E*+I*)/I*.
matched_guide(par) = begin
    eq = equilibrium(; par...)
    fsmarkov(Expos=>eq.E, Infec=>eq.I, (Expos,Infec)=>par.σ*(eq.E+eq.I)/eq.I)
end

## n independent log-likelihood estimates, with mean seconds per run
logliks(tr, n; kwargs...) = begin
    pfilter(tr; Np=50, kwargs...)
    t = @elapsed ll = [logLik(pfilter(tr; kwargs...)) for _ ∈ 1:n]
    ll, t/n
end

summarize(ll) = begin
    fin = filter(isfinite, ll)
    isempty(fin) && return (lme=-Inf, se=NaN, sd=NaN, nfinite=0)
    e = logmeanexp(ll, se=true)
    (lme=e.est, se=e.se, sd=std(fin), nfinite=length(fin))
end

write_csv(path, rows::Vector{<:NamedTuple}) = open(path, "w") do io
    println(io, join(keys(first(rows)), ","))
    for r ∈ rows
        println(io, join((v isa AbstractString ? "\"$v\"" : string(v) for v ∈ values(r)), ","))
    end
end

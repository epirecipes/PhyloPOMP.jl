## Shared helpers for the scalability experiments (include from scripts).
using PhyloPOMP
using PhyloPOMP: NaiveSEIR, SoftSEIR, GuidedSEIR, HardSEIR, ExactSEIR, SimSEIR
using Random, Statistics, Printf
import PartiallyObservedMarkovProcesses as POMP

const RESULTS = joinpath(@__DIR__, "results")
const LAB = normpath(joinpath(@__DIR__, "..", "..", "lab"))

## default guide process used throughout the package tests
default_guide() = fsmarkov(GuidedSEIR.Demes.Expos=>0.1, GuidedSEIR.Demes.Infec=>1,
                           (GuidedSEIR.Demes.Expos, GuidedSEIR.Demes.Infec)=>1)
guide_for(mod) = fsmarkov(mod.Demes.Expos=>0.1, mod.Demes.Infec=>1, (mod.Demes.Expos, mod.Demes.Infec)=>1)

## build the POMP object of a given filter variant
make_pomp(variant::Symbol, g; kw...) =
    variant === :naive  ? NaiveSEIR.filter_pomp(g; kw...) :
    variant === :soft   ? SoftSEIR.filter_pomp(g, guide_for(SoftSEIR); kw...) :
    variant === :guided ? GuidedSEIR.filter_pomp(g, guide_for(GuidedSEIR); kw...) :
    variant === :hard   ? HardSEIR.filter_pomp(g, guide_for(HardSEIR); kw...) :
    error("unknown variant $variant")

## n independent pfilter log-likelihood estimates
pf_logliks(p, Np, n) = [logLik(pfilter(p; Np)) for _ ∈ 1:n]

## summarize replicate estimates: logmeanexp with jackknife SE, plus sd of log-lik
summarize(ll) = begin
    fin = filter(isfinite, ll)
    lme = isempty(fin) ? -Inf : logmeanexp(ll, se=true)
    (lme = isempty(fin) ? -Inf : lme.est, se = isempty(fin) ? NaN : lme.se,
     sd = length(fin) > 1 ? std(fin) : NaN, nfinite = length(fin), n = length(ll))
end

## minimal CSV writer (avoid extra dependencies)
write_csv(path, rows::Vector{<:NamedTuple}) = open(path, "w") do io
    ks = keys(first(rows))
    println(io, join(ks, ","))
    for r ∈ rows
        println(io, join((v isa AbstractString ? "\"$v\"" : string(v) for v ∈ values(r)), ","))
    end
end

## minimal JSON for event export (Vector of NamedTuples / Dicts of numbers, strings, vectors)
tojson(x::AbstractString) = "\"$x\""
tojson(x::Real) = isfinite(x) ? repr(x) : "null"
tojson(x::Symbol) = tojson(String(x))
tojson(x::AbstractVector) = "[" * join(map(tojson, x), ",") * "]"
tojson(x::NamedTuple) = "{" * join(("\"$k\":" * tojson(v) for (k,v) ∈ pairs(x)), ",") * "}"
tojson(x::Tuple) = tojson(collect(x))

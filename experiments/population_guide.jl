## A deterministic population forecast supplies interval-specific lineage rates.
## Only the colour proposal changes; FastSEIR still supplies every target weight
## and simulates the population exactly.
module PopulationGuide

using Random: AbstractRNG, default_rng
using PhyloPOMP
using PhyloPOMP: Root
const F = PhyloPOMP.FastSEIR

struct ForecastTree{P<:NamedTuple}
    intervals::Vector{F.Tree}
    params::P
    initial::NTuple{4,Int}
    population::Matrix{Float64}
end

drift(x, p) = begin
    S, E, I, R = x
    T, P, C, W = p.β*S*I/p.pop, p.σ*E, p.γ*I, p.ω*R
    (W-T,T-P,P-C,C-W)
end

rk4(x, h, p) = begin
    a = drift(x,p)
    b = drift(ntuple(i -> x[i]+h*a[i]/2,4),p)
    c = drift(ntuple(i -> x[i]+h*b[i]/2,4),p)
    d = drift(ntuple(i -> x[i]+h*c[i],4),p)
    ntuple(i -> x[i]+h*(a[i]+2b[i]+2c[i]+d[i])/6,4)
end

forecast(tr, initial, p, step) = begin
    @assert isfinite(step) && step > 0
    hmax = min(step,0.1/max(p.β,p.σ,p.γ,p.ω,1e-8))
    x = map(Float64,initial)
    t = tr.time[1]
    pop = zeros(4,length(tr.time))
    for k ∈ eachindex(tr.time)
        midpoint = (tr.time[k]+tr.tend[k])/2
        while t < midpoint
            h = min(hmax,midpoint-t)
            x = rk4(x,h,p)
            @assert all(isfinite,x) && all(≥(0),x) "nonphysical forecast; reduce step"
            t = h == midpoint-t ? midpoint : t+h
        end
        pop[:,k] .= x
    end
    pop
end

"""
    ForecastTree(g; step=0.02, model_params...)

Compile a forward SEIRS mean-population forecast into a colour guide. Lineage
rates are held constant on each genealogical interval, evaluated at the forecast
midpoint: E->I = sigma*(E+1)/(I+1), I->E = beta*S*(I+1)/(pop*(E+1)).
Unit pseudocounts regularize small compartments and a positive rate floor keeps
proposal support. These approximations affect the guide, not the target.

The usual deterministic genealogy messages supply the child probabilities and
lineage hazards. Non-parent ratios are frozen mid-interval as in E23; the parent
of the next event is guided at its actual jump time. Population dynamics run
forward; no reverse-time stochastic simulation is introduced.

Parameters are stored with the compiled guide and cannot be overridden in
`pfilter_forecast`. Only non-destructive sampling is currently supported.
"""
ForecastTree(
    g::Genealogy; step::Real=0.02,
    β=4.0, σ=1.0, γ=1.0, ω=1.0, ψ=0.02, χ=0.0,
    pop=100, S0=0.9, E0=0.0, I0=0.02, R0=0.08,
) = begin
    @assert χ == 0 "the population forecast currently requires chi=0"
    p = map(Float64,(;β,σ,γ,ω,ψ,χ,pop))
    @assert all(isfinite,p) && all(≥(0),p) && pop > 0
    base = F.Tree(g)
    initial = barycentric(Int64,(S0,E0,I0,R0),pop)
    trajectory = forecast(base,initial,p,step)
    M = length(base.time)
    c1, c2 = zeros(length(base.lins)), zeros(length(base.lins))
    toI, toE = similar(c1), similar(c1)
    present = [zeros(2,length(c)) for c ∈ base.chillins]
    intervals = Vector{F.Tree}(undef,M)
    messages = zeros(2,Int(g.nsample))
    for k ∈ M:-1:1
        S, E, I, _ = trajectory[:,k]
        a = max(p.σ*(E+1)/(I+1),1e-8)
        b = max(p.β*S*(I+1)/(p.pop*(E+1)),1e-8)
        κ = a+b
        πE, πI = b/κ, a/κ
        A = [1.0 -πI; 1.0 πE]
        Δ = base.tend[k]-base.time[k]
        for j ∈ base.ptr[k]:base.ptr[k+1]-1
            lin = base.lins[j]
            hE, hI = messages[:,lin]
            c1[j], c2[j] = πE*hE+πI*hI, hI-hE
            midE, midI = F.guideprob(A,c1[j],c2[j],exp(-κ*Δ/2))
            toI[j], toE[j] = F.ratio(midI,midE), F.ratio(midE,midI)
            messages[:,lin] .= F.guideprob(A,c1[j],c2[j],exp(-κ*Δ))
        end
        for (c,lin) ∈ enumerate(base.chillins[k])
            hE, hI = messages[:,lin]
            total = πE*hE+πI*hI
            @assert total > 0 "zero child guide message"
            present[k][:,c] .= (πE*hE/total,πI*hI/total)
        end
        messages[:,base.parlin[k]] .= base.type[k] == Root ? (1.0,1.0) : (0.0,1.0)
        intervals[k] = F.Tree(
            base.time,base.tend,base.type,base.parlin,base.chillins,base.ptr,base.lins,
            base.nwords,true,present,c1,c2,A,κ,toI,toE,base.focal,
        )
    end
    ForecastTree(intervals,p,initial,trajectory)
end

"""
    pfilter_forecast(tr::ForecastTree; Np=1, threaded=true, rng=default_rng())

Exact-population particle filter using the compiled, parameter-bound forecast
guide. All population, colour and observed-event importance weights are reused
from FastSEIR. Guide compilation can be amortized over likelihood replicates.
"""
pfilter_forecast(
    tr::ForecastTree; Np::Integer=1, threaded::Bool=true,
    rng::AbstractRNG=default_rng(),
) = begin
    @assert Np > 0
    first = tr.intervals[1]
    M = length(first.time)
    P = F.Particles(Np,first.nwords); Q = F.Particles(Np,first.nwords)
    S, E, I, R = tr.initial
    P.S .= S; P.E .= E; P.I .= I; P.R .= R
    p = tr.params
    W = fill(1/Np,Np)
    ess = fill(Float64(Np),M); cll = zeros(M); zerofrac = zeros(M)
    ancestors = zeros(Int,Np)
    seed = rand(rng,UInt64)
    loglik = 0.0
    nevents = 0
    k0 = findlast(==(first.time[1]),first.time)
    for j ∈ 1:k0
        F.foreach_particle(P,seed,-j,threaded) do P, i, rng
            F.singular_part!(first,P,i,j,p,rng; guided=true)
            0
        end
    end
    for k ∈ k0:M
        current = tr.intervals[k]
        nevents += F.foreach_particle(P,seed,k,threaded) do P, i, rng
            isfinite(P.ll[i]) || return 0
            n = F.regular_part!(current,P,i,k,p,rng; proposal=:local,guided=true)
            k < M && F.singular_part!(current,P,i,k+1,p,rng; guided=true)
            n
        end
        mx = maximum(P.ll)
        isfinite(mx) || return F.FastPfilterd(-Inf,ess,cll,zerofrac,nevents)
        W .*= exp.(P.ll .- mx)
        total = sum(W)
        cll[k] = mx+log(total)
        loglik += cll[k]
        W ./= total
        ess[k] = 1/sum(abs2,W)
        zerofrac[k] = count(iszero,W)/Np
        F.systematic!(ancestors,W,rng)
        for (j,a) ∈ enumerate(ancestors)
            F.copyparticle!(Q,j,P,a)
        end
        P, Q = Q, P
        W .= 1/Np
    end
    F.FastPfilterd(loglik,ess,cll,zerofrac,nevents)
end

end

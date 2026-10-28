## Time-varying, support-correct "hard" proposals for the FastSEIR regular part.
##
## Proposal family (per particle, interval k): every outcome o of every population
## jump u is a separate reaction with rate
##     q_o(t,x) = α_u(x) φ_o(x) r_o(t) e_u(t)
## where φ_o is the target boost (locally optimal base weights), r_o(t) the guide's
## relative hazard for a lineage colour change (1 for "no change"), and
## e_u(t) = exp(clamp(g(t)·ν_u)) an optional forward-ODE population twist.
## Boost at a jump: α_u φ_o / q_o = 1/(r_o e_u); decay λ̃ = λ + Σα − Σq.
##
## Two exact implementations of the same family:
##   `regular_frozen!`: rates frozen at the last jump (piecewise-constant hazards),
##   `regular_pdmp!`:   exact time-varying hazards, jump times sampled by
##                      PiecewiseDeterministicMarkovProcesses.jl (CHV algorithm);
##                      segment integrals of λ̃ by Gauss–Legendre quadrature.
module PDMPProposals

using PhyloPOMP
using PhyloPOMP: FastSEIR
using PhyloPOMP.FastSEIR: Tree, Particles, getbit, setbit!, clrbit!, relhaz, flow,
    ROOT, SAMPLE, BRANCH
using PiecewiseDeterministicMarkovProcesses
const PDMP = PiecewiseDeterministicMarkovProcesses
using OrdinaryDiffEq: Tsit5
using Random: AbstractRNG, randexp
using LinearAlgebra: SymTridiagonal, eigen

Base.@kwdef struct HardCfg
    guided::Bool = true
    twist::Bool = false
    horizon::Int = 2
    cmax::Float64 = 2.0
    rmax::Float64 = 1e8
end

const EPS = 1e-9        # weight-neutral self-jump keeps the total rate positive (CHV)
const NU = ((-1, 1, 0, 0), (0, -1, 1, 0), (0, 0, -1, 1), (1, 0, 0, -1))

## ---------------------------------------------------------------- twist

## deterministic look-ahead V(x) from the start of interval k through `h` events
lookahead_value(tr::Tree, x::NTuple{4,Float64}, lI::Float64, k::Int, p, h::Int) = begin
    (; β, ψ, χ, pop) = p
    M = length(tr.time)
    s = 0.0
    for m ∈ k:min(k+h-1, M)
        x, dec = flow(x, tr.tend[m]-tr.time[m], p)
        s -= dec
        m < M || break
        S, E, I, R = x
        typ = tr.type[m+1]
        if typ == BRANCH
            s += log(β*max(S, 0.5)/(pop*(E+1)))
            x = (max(S-1, 0.0), E+1, I, R); lI += 1
        elseif typ == SAMPLE && tr.nch[m+1] == 0
            s += log(ψ*max(I-lI+1, 0.5) + χ*max(I, 0.5)); lI -= 1
        elseif typ == SAMPLE
            s += log(ψ)
        end
    end
    s
end

## gradient of V by central differences (unit steps)
twist_gradient(tr, P, i, k, p, h) = begin
    x = (Float64(P.S[i]), Float64(P.E[i]), Float64(P.I[i]), Float64(P.R[i]))
    lI = Float64(P.lI[i])
    ntuple(4) do d
        xp = ntuple(j -> j == d ? x[j]+1 : x[j], 4)
        xm = ntuple(j -> j == d ? max(x[j]-1, 0.0) : x[j], 4)
        (lookahead_value(tr, xp, lI, k, p, h) - lookahead_value(tr, xm, lI, k, p, h))/(xp[d]-xm[d])
    end
end

@inline twistfac(cfg, g0, tk, t, p, u) = begin
    cfg.twist || return 1.0
    ## the remaining sampling-decay horizon shrinks as t advances: ∂V/∂I grows by ψ+χ
    gI = g0[3] + (p.ψ + p.χ)*(t - tk)
    η = g0[1]*NU[u][1] + g0[2]*NU[u][2] + gI*NU[u][3] + g0[4]*NU[u][4]
    exp(clamp(η, -cfg.cmax, cfg.cmax))
end

## ---------------------------------------------------------------- rates

## Fill `rate` (length 5+2n; n lineages on interval k) for state x at time t.
## Layout: 1 Trans/no-change, 2 Prog/no-change, 3 Recov, 4 Wane,
## 4+j Trans swapping lineage j (I→E), 4+n+j Prog swapping lineage j (E→I),
## 5+2n weight-neutral self-jump.  Returns (A, λ): target out-rate and decay.
rates!(rate, tr::Tree, P::Particles, i::Int, k::Int, x::NTuple{4,Int}, t::Float64,
       p, cfg::HardCfg, g0, tk) = begin
    (; β, σ, γ, ω, ψ, χ, pop) = p
    S, E, I, R = x
    lE, lI = P.lE[i], P.lI[i]
    aT = β*S*I/pop; aP = σ*E; aR = I > lI ? γ*I : 0.0; aW = ω*R
    λ = (ψ+χ)*I + (I > lI ? 0.0 : γ*I)
    eT = twistfac(cfg, g0, tk, t, p, 1); eP = twistfac(cfg, g0, tk, t, p, 2)
    eR = twistfac(cfg, g0, tk, t, p, 3); eW = twistfac(cfg, g0, tk, t, p, 4)
    φ0T = (E+1-lE)/(E+1)
    φbT = I ≥ 1 ? (I-lI+1)/((E+1)*I) : 0.0
    φ0P = E-1 ≥ lE ? (I+1-lI)/(I+1) : 0.0
    φbP = 1.0/(I+1)
    rate[1] = aT*φ0T*eT
    rate[2] = aP*φ0P*eP
    rate[3] = aR*eR
    rate[4] = aW*eW
    j0 = tr.ptr[k]; n = tr.ptr[k+1]-j0
    e = cfg.guided && tr.guided ? exp(-tr.κ*(tr.tend[k]-t)) : 1.0
    @inbounds for jj ∈ 1:n
        j = j0+jj-1
        b = tr.lins[j]
        if getbit(P.col, i, b)
            r = cfg.guided && tr.guided ? min(relhaz(tr, j, e, false), cfg.rmax) : 1.0
            rate[4+jj] = aT*φbT*r*eT
            rate[4+n+jj] = 0.0
        else
            r = cfg.guided && tr.guided ? min(relhaz(tr, j, e, true), cfg.rmax) : 1.0
            rate[4+jj] = 0.0
            rate[4+n+jj] = aP*φbP*r*eP
        end
    end
    rate[5+2n] = EPS
    aT + aP + aR + aW + EPS, λ
end

## log boost of reaction `ev` given the (pre-jump) rates at the jump time
logboost(rate, ev, n, x, P, i, p) = begin
    (; β, σ, γ, ω, pop) = p
    S, E, I, R = x
    lE, lI = P.lE[i], P.lI[i]
    target = if ev == 1
        β*S*I/pop*(E+1-lE)/(E+1)
    elseif ev == 2
        σ*E*(E-1 ≥ lE ? (I+1-lI)/(I+1) : 0.0)
    elseif ev == 3
        I > lI ? γ*I : 0.0
    elseif ev == 4
        ω*R
    elseif ev == 5+2n
        EPS                # weight-neutral self-jump
    elseif ev ≤ 4+n
        β*S*I/pop*(I ≥ 1 ? (I-lI+1)/((E+1)*I) : 0.0)
    else
        σ*E/(I+1)
    end
    log(target) - log(rate[ev])
end

## apply reaction ev to state x and colouring; returns new x
apply(ev, n, x, tr, P, i, k) = begin
    S, E, I, R = x
    if ev == 1
        (S-1, E+1, I, R)
    elseif ev == 2
        (S, E-1, I+1, R)
    elseif ev == 3
        (S, E, I-1, R+1)
    elseif ev == 4
        (S+1, E, I, R-1)
    elseif ev == 5+2n
        x
    elseif ev ≤ 4+n
        b = tr.lins[tr.ptr[k]+ev-5]
        clrbit!(P.col, i, b); P.lI[i] -= 1; P.lE[i] += 1
        (S-1, E+1, I, R)
    else
        b = tr.lins[tr.ptr[k]+ev-5-n]
        setbit!(P.col, i, b); P.lE[i] -= 1; P.lI[i] += 1
        (S, E-1, I+1, R)
    end
end

## ---------------------------------------------------------------- quadrature

const GL = let n = 16
    b = [m/sqrt(4m^2-1) for m ∈ 1:n-1]
    F = eigen(SymTridiagonal(zeros(n), b))
    ((F.values .+ 1) ./ 2, F.vectors[1,:].^2)
end

## ∫_a^b λ̃(s) ds for fixed state x (4 panels × 16-point Gauss–Legendre)
decay_integral(buf, tr, P, i, k, x, a, b, p, cfg, g0, tk) = begin
    b > a || return 0.0
    xs, ws = GL
    tot = 0.0
    np = 4; h = (b-a)/np
    for q ∈ 0:np-1, (u, w) ∈ zip(xs, ws)
        s = a + h*(q+u)
        A, λ = rates!(buf, tr, P, i, k, x, s, p, cfg, g0, tk)
        tot += w*h*(λ + A - sum(buf))
    end
    tot
end

## ---------------------------------------------------------------- frozen hazards

regular_frozen!(tr::Tree, P::Particles, i::Int, k::Int, t::Float64, tf::Float64,
                p, opt, rng::AbstractRNG; cfg::HardCfg) = begin
    n = tr.ptr[k+1]-tr.ptr[k]
    buf = Vector{Float64}(undef, 5+2n)
    g0 = cfg.twist ? twist_gradient(tr, P, i, k, p, cfg.horizon) : (0.0, 0.0, 0.0, 0.0)
    tk = t
    x = (P.S[i], P.E[i], P.I[i], P.R[i])
    logw = P.logw[i]
    nev = 0
    while true
        A, λ = rates!(buf, tr, P, i, k, x, t, p, cfg, g0, tk)
        Q = sum(buf)
        dec = λ + A - Q
        step = randexp(rng)/Q
        if t + step ≥ tf
            logw -= dec*(tf-t); break
        end
        logw -= dec*step
        t += step
        u = rand(rng)*Q; ev = 1
        while u > buf[ev] && ev < length(buf)
            u -= buf[ev]; ev += 1
        end
        logw += logboost(buf, ev, n, x, P, i, p)
        x = apply(ev, n, x, tr, P, i, k)
        nev += 1
    end
    P.S[i], P.E[i], P.I[i], P.R[i] = x
    P.logw[i] = logw
    nev
end

## ---------------------------------------------------------------- PDMP (CHV)

mutable struct Ctx{TP}
    tr::Tree
    P::Particles
    i::Int
    k::Int
    p::TP
    cfg::HardCfg
    g0::NTuple{4,Float64}
    tk::Float64
    n::Int
    buf::Vector{Float64}
    tlast::Float64
    logw::Float64
end

pdmp_R!(rate, xc, xd, c::Ctx, t, issum) = begin
    x = (xd[1], xd[2], xd[3], xd[4])
    rates!(rate, c.tr, c.P, c.i, c.k, x, t, c.p, c.cfg, c.g0, c.tk)
    issum ? (sum(rate), 0.0) : (0.0, 0.0)
end

pdmp_F!(dxc, xc, xd, c::Ctx, t) = (dxc[1] = 0.0; nothing)

pdmp_Delta!(xc, xd, c::Ctx, t, ev) = begin
    x = (xd[1], xd[2], xd[3], xd[4])
    ## weight of the segment since the previous jump, then the jump boost
    c.logw -= decay_integral(c.buf, c.tr, c.P, c.i, c.k, x, c.tlast, t, c.p, c.cfg, c.g0, c.tk)
    rates!(c.buf, c.tr, c.P, c.i, c.k, x, t, c.p, c.cfg, c.g0, c.tk)
    c.logw += logboost(c.buf, ev, c.n, x, c.P, c.i, c.p)
    x2 = apply(ev, c.n, x, c.tr, c.P, c.i, c.k)
    xd[1], xd[2], xd[3], xd[4] = x2
    c.tlast = t
    nothing
end

regular_pdmp!(tr::Tree, P::Particles, i::Int, k::Int, t::Float64, tf::Float64,
              p, opt, rng::AbstractRNG; cfg::HardCfg, reltol = 1e-9, abstol = 1e-11) = begin
    tf > t || return 0
    n = tr.ptr[k+1]-tr.ptr[k]
    g0 = cfg.twist ? twist_gradient(tr, P, i, k, p, cfg.horizon) : (0.0, 0.0, 0.0, 0.0)
    c = Ctx(tr, P, i, k, p, cfg, g0, t, n, Vector{Float64}(undef, 5+2n), t, P.logw[i])
    xd0 = [P.S[i], P.E[i], P.I[i], P.R[i]]
    prob = PDMP.PDMPProblem(pdmp_F!, pdmp_R!, pdmp_Delta!, 5+2n, [0.0], xd0, c, (t, tf))
    res = PDMP.solve(prob, PDMP.CHV(Tsit5()); save_positions = (false, false), reltol, abstol)
    xd = prob.caract.xd
    x = (xd[1], xd[2], xd[3], xd[4])
    c.logw -= decay_integral(c.buf, tr, P, i, k, x, c.tlast, tf, p, cfg, g0, t)
    P.S[i], P.E[i], P.I[i], P.R[i] = x
    P.logw[i] = c.logw
    res.njumps
end

"""
    options(kind; cfg, base...)

FastSEIR options using the frozen-hazard (`:frozen`) or PDMP (`:pdmp`) regular part.
"""
options(kind::Symbol; cfg::HardCfg = HardCfg(), base...) = FastSEIR.Options(;
    threaded = false, base...,
    regular = kind === :pdmp ?
        ((tr, P, i, k, t, tf, p, opt, rng) -> regular_pdmp!(tr, P, i, k, t, tf, p, opt, rng; cfg)) :
        ((tr, P, i, k, t, tf, p, opt, rng) -> regular_frozen!(tr, P, i, k, t, tf, p, opt, rng; cfg)))

end

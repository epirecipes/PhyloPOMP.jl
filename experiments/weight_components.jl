## Instrumented copy of FastSEIR's propagation that splits each particle's
## log weight over an interval into
##   phi   Σ log Σφ over population events (the exact total color weight; depends on counts),
##         or, with weighted rates, its compensator -∫ λ(1-Σφ) dt
##   is    Σ log of the importance ratio of the color choice (proposal vs exact weights)
##   decay -∫ ((ψ+χ)I + γI 1[I ≤ ℓI]) dt
##   event log weight of the closing genealogical event (-Inf if incompatible)
## and records why incompatible particles failed.
module WeightDiag

using Random: AbstractRNG, Xoshiro, randexp
using PhyloPOMP
using PhyloPOMP: Root, Node, Sample, Name
using PhyloPOMP.FastSEIR: Tree, Particles, isinfec, infec!, expos!, swap_weight,
    singular_part!, copyparticle!, systematic!, foreach_particle

const PHI, IS, DECAY, EVENT = 1, 2, 3, 4

color_change(tr::Tree, P::Particles, i, k, t, toI::Bool, rng; guided) = begin
    E, I, ellE, ellI = P.E[i], P.I[i], P.ellE[i], P.ellI[i]
    φ0, φb = toI ?
        (@indicator(E > ellE, (I+1-ellI)/(I+1)), 1/(I+1)) :
        ((E+1-ellE)/(E+1), (I+1-ellI)/((E+1)*I))
    e = guided ? exp(-tr.κ*(tr.tend[k]-t)) : 1.0
    lins = tr.ptr[k]:(tr.ptr[k+1]-1)
    total = φ0
    nmov = 0
    for j ∈ lins
        total += swap_weight(tr,P,i,j,toI,φb,e,guided)
        nmov += isinfec(P,i,tr.lins[j]) ≠ toI
    end
    Σφ = φ0 + nmov*φb
    u = rand(rng)*total
    u < φ0 && return zero(Name), log(Σφ), log(total/Σφ)
    u -= φ0
    jb, wb = 0, 0.0
    for j ∈ lins
        w = swap_weight(tr,P,i,j,toI,φb,e,guided)
        if w > 0
            jb, wb = j, w
            u < w && break
            u -= w
        end
    end
    tr.lins[jb], log(Σφ), log(φb*total/(wb*Σφ))
end

regular_part!(C, tr::Tree, P::Particles, i, k, p, rng; guided, weighted) = begin
    (; β, σ, γ, ω, ψ, χ, pop) = p
    t = tr.time[k]
    tf = tr.tend[k]
    while true
        S, E, I, R, ellE, ellI = P.S[i], P.E[i], P.I[i], P.R[i], P.ellE[i], P.ellI[i]
        transmission = β*S*I/pop
        progression = σ*E
        ## weighted rates: simulate at λΣφ; the difference λ(1-Σφ) joins the weight
        comp = 0.0
        if weighted
            ΦT = I > 0 ? ((E+1-ellE)*I + ellI*(I+1-ellI))/((E+1)*I) : 1.0
            ΦP = (@indicator(E > ellE, I+1-ellI) + ellE)/(I+1)
            comp = transmission*(1-ΦT) + progression*(1-ΦP)
            transmission *= ΦT; progression *= ΦP
        end
        recovery = @indicator(I > ellI, γ*I)
        waning = ω*R
        decay = (ψ+χ)*I + @indicator(I ≤ ellI, γ*I)
        rate = transmission+progression+recovery+waning
        step = randexp(rng)/rate
        if t+step ≥ tf
            C[DECAY,i] -= decay*(tf-t)
            C[PHI,i] -= comp*(tf-t)
            break
        end
        C[DECAY,i] -= decay*step
        C[PHI,i] -= comp*step
        t += step
        u = rand(rng)*rate
        if u < transmission+progression
            toI = u ≥ transmission
            b, lphi, lis = color_change(tr,P,i,k,t,toI,rng;guided)
            weighted || (C[PHI,i] += lphi)
            C[IS,i] += lis
            if toI
                P.E[i] -= 1; P.I[i] += 1
                b > 0 && (infec!(P,i,b); P.ellE[i] -= 1; P.ellI[i] += 1)
            else
                P.S[i] -= 1; P.E[i] += 1
                b > 0 && (expos!(P,i,b); P.ellI[i] -= 1; P.ellE[i] += 1)
            end
        elseif u < transmission+progression+recovery
            P.I[i] -= 1; P.R[i] += 1
        else
            P.R[i] -= 1; P.S[i] += 1
        end
    end
end

## failure causes: 1 parent not infectious, 2 no susceptible, 3 no room for root
failure(tr::Tree, P::Particles, i, k) =
    tr.type[k] == Root ? ((P.E[i] > P.ellE[i] || P.I[i] > P.ellI[i]) ? 0 : 3) :
    !isinfec(P,i,tr.parlin[k]) ? 1 :
    (tr.type[k] == Node && P.S[i] == 0) ? 2 : 0

wvar(x) = (m = sum(x)/length(x); sum(abs2, x .- m)/length(x))
essfrac(lw) = (isempty(lw) || !all(isfinite,lw)) ? NaN :
    (w = exp.(lw .- maximum(lw)); sum(w)^2/sum(abs2,w)/length(w))

"""
    diagnose(tr; Np, params..., guided) -> Vector of per-interval NamedTuples

Filter with resampling after every event; for each interval report the variance
across particles (with finite weight) of each log-weight component, the ESS
fraction each component alone would give, and the failure fractions.
"""
diagnose(
    tr::Tree; Np::Integer,
    β = 4.0, σ = 1.0, γ = 1.0, ω = 1.0, ψ = 0.02, χ = 0.0,
    pop = 100, S0 = 0.9, E0 = 0.0, I0 = 0.02, R0 = 0.08,
    guided::Bool = true, weighted::Bool = false, seed::UInt64 = UInt64(1),
) = begin
    useguide = guided && tr.guided
    p = map(Float64, (;β,σ,γ,ω,ψ,χ,pop))
    M = length(tr.time)
    P = Particles(Np,tr.nwords); Q = Particles(Np,tr.nwords)
    S, E, I, R = barycentric(Int64,(S0,E0,I0,R0),pop)
    P.S .= S; P.E .= E; P.I .= I; P.R .= R
    C = zeros(4,Np)
    cause = zeros(Int,Np)
    ancestors = Vector{Int}(undef,Np)
    rng = Xoshiro(seed)
    rows = NamedTuple[]
    k0 = findlast(==(tr.time[1]),tr.time)
    for j ∈ 1:k0
        foreach_particle(P,seed,-j,true) do P, i, rng
            singular_part!(tr,P,i,j,p,rng;guided=useguide); 0
        end
    end
    for k ∈ k0:M-1
        C .= 0.0; cause .= 0
        foreach_particle(P,seed,k,true) do P, i, rng
            regular_part!(C,tr,P,i,k,p,rng;guided=useguide,weighted)
            cause[i] = failure(tr,P,i,k+1)
            before = P.ll[i]
            singular_part!(tr,P,i,k+1,p,rng;guided=useguide)
            C[EVENT,i] = P.ll[i] - before
            0
        end
        ok = findall(isfinite, @view C[EVENT,:])
        isempty(ok) && break
        total = vec(sum(@view(C[:,ok]),dims=1))
        lin = tr.ptr[k+1]-tr.ptr[k]
        push!(rows, (interval=k, closing=string(tr.type[k+1]), length=tr.tend[k]-tr.time[k],
            lineages=lin, fail_parentE=count(==(1),cause)/Np, fail_S0=count(==(2),cause)/Np,
            fail_root=count(==(3),cause)/Np,
            var_phi=wvar(C[PHI,ok]), var_is=wvar(C[IS,ok]), var_decay=wvar(C[DECAY,ok]),
            var_event=wvar(C[EVENT,ok]), var_total=wvar(total),
            ess_phi=essfrac(C[PHI,ok]), ess_is=essfrac(C[IS,ok]), ess_decay=essfrac(C[DECAY,ok]),
            ess_event=essfrac(C[EVENT,ok]), ess_total=essfrac(total)*length(ok)/Np))
        lw = fill(-Inf,Np); lw[ok] .= total
        w = exp.(lw .- maximum(lw)); w ./= sum(w)
        systematic!(ancestors,w,rng)
        for (j,a) ∈ enumerate(ancestors)
            copyparticle!(Q,j,P,a)
        end
        P, Q = Q, P
    end
    rows
end

end

## Experimental "weighted-rate" proposal for FastSEIR.
##
## FastSEIR simulates transmissions and progressions at their prior rates λ and,
## at each one, multiplies the weight by Σφ, the total exact weight of the color
## changes that may accompany it (a function of E, I, ℓE, ℓI).  Here they are
## simulated instead at the weighted rates λΣφ, the color change is chosen as
## before, and the difference λ(1-Σφ) joins the decay.  The importance weight of
## a jump is then only the ratio of the guided to the exact color-choice
## probabilities.  The target, and hence the likelihood, is unchanged.
module WFilter

using Random: AbstractRNG, default_rng, randexp
using PhyloPOMP
using PhyloPOMP: Name, Root
using PhyloPOMP.FastSEIR: Tree, Particles, FastPfilterd, isinfec, infec!, expos!,
    swap_weight, guideprob, singular_part!, copyparticle!, systematic!, foreach_particle

## total exact color weights accompanying a transmission and a progression
Φtrans(E, I, ellE, ellI) = I > 0 ? ((E+1-ellE)*I + ellI*(I+1-ellI))/((E+1)*I) : 1.0
Φprog(E, I, ellE, ellI) = (@indicator(E > ellE, I+1-ellI) + ellE)/(I+1)

## as FastSEIR.color_change, but the log weight is relative to Σφ
color_change(tr::Tree, P::Particles, i, k, t, toI::Bool, Σφ, rng; guided) = begin
    E, I, ellE, ellI = P.E[i], P.I[i], P.ellE[i], P.ellI[i]
    φ0, φb = toI ?
        (@indicator(E > ellE, (I+1-ellI)/(I+1)), 1/(I+1)) :
        ((E+1-ellE)/(E+1), (I+1-ellI)/((E+1)*I))
    e = guided ? exp(-tr.κ*(tr.tend[k]-t)) : 1.0
    lins = tr.ptr[k]:(tr.ptr[k+1]-1)
    total = φ0
    for j ∈ lins
        total += swap_weight(tr,P,i,j,toI,φb,e,guided)
    end
    u = rand(rng)*total
    u < φ0 && return zero(Name), log(total/Σφ)
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
    tr.lins[jb], log(φb*total/(wb*Σφ))
end

regular_part!(tr::Tree, P::Particles, i, k, p, rng; guided) = begin
    (; β, σ, γ, ω, ψ, χ, pop) = p
    t = tr.time[k]
    tf = tr.tend[k]
    ll = P.ll[i]
    nevent = 0
    while true
        S, E, I, R, ellE, ellI = P.S[i], P.E[i], P.I[i], P.R[i], P.ellE[i], P.ellI[i]
        λT, λP = β*S*I/pop, σ*E
        ΦT, ΦP = Φtrans(E,I,ellE,ellI), Φprog(E,I,ellE,ellI)
        transmission = λT*ΦT
        progression = λP*ΦP
        recovery = @indicator(I > ellI, γ*I)
        waning = ω*R
        decay = (ψ+χ)*I + @indicator(I ≤ ellI, γ*I) + λT*(1-ΦT) + λP*(1-ΦP)
        rate = transmission+progression+recovery+waning
        step = randexp(rng)/rate
        if t+step ≥ tf
            ll -= decay*(tf-t)
            break
        end
        ll -= decay*step
        t += step
        nevent += 1
        u = rand(rng)*rate
        if u < transmission
            b, lw = color_change(tr,P,i,k,t,false,ΦT,rng;guided)
            ll += lw
            P.S[i] -= 1; P.E[i] += 1
            if b > 0
                expos!(P,i,b); P.ellI[i] -= 1; P.ellE[i] += 1
            end
        elseif u < transmission+progression
            b, lw = color_change(tr,P,i,k,t,true,ΦP,rng;guided)
            ll += lw
            P.E[i] -= 1; P.I[i] += 1
            if b > 0
                infec!(P,i,b); P.ellE[i] -= 1; P.ellI[i] += 1
            end
        elseif u < transmission+progression+recovery
            P.I[i] -= 1; P.R[i] += 1
        else
            P.R[i] -= 1; P.S[i] += 1
        end
    end
    P.ll[i] = ll
    nevent
end

## Twisted (Doob-style) regular part.  The guide's ratio r = h(other)/h(own) of
## each lineage multiplies the weighted rate of the color change that moves it,
## rather than only its probability given a population event.  The ratios are
## frozen between jumps (and refreshed on a geometric grid towards the end of
## the interval while the parent of the closing event is exposed), so the
## intensity is predictable and the weight exact: a color change of lineage b
## contributes 1/r_b and the difference of total rates joins the decay.  With
## `tilt = false` the rates are also divided by Σφ (which then multiplies the
## weight at each jump, as in FastSEIR), so only the guide changes the rates.
twisted_part!(tr::Tree, P::Particles, i, k, p, rng; guided, tilt) = begin
    (; β, σ, γ, ω, ψ, χ, pop) = p
    t = tr.time[k]
    tf = tr.tend[k]
    lins = tr.ptr[k]:(tr.ptr[k+1]-1)
    par = k < length(tr.time) && tr.type[k+1] ≠ Root ? tr.parlin[k+1] : zero(Name)
    ll = P.ll[i]
    nevent = 0
    while true
        S, E, I, R, ellE, ellI = P.S[i], P.E[i], P.I[i], P.R[i], P.ellE[i], P.ellI[i]
        λT, λP = β*S*I/pop, σ*E
        φ0T = (E+1-ellE)/(E+1)
        φbT = I > 0 ? (I+1-ellI)/((E+1)*I) : 0.0
        φ0P = @indicator(E > ellE, (I+1-ellI)/(I+1))
        φbP = 1/(I+1)
        e = guided ? exp(-tr.κ*(tf-t)) : 1.0
        ΣrI, ΣrE = 0.0, 0.0
        for j ∈ lins
            r = ratio(tr,P,i,j,e,guided)
            isinfec(P,i,tr.lins[j]) ? (ΣrI += r) : (ΣrE += r)
        end
        ## without `tilt`, divide by Σφ so that, unguided, the rates are the prior's
        ΦT = tilt ? 1.0 : Φtrans(E,I,ellE,ellI)
        ΦP = tilt || E == 0 ? 1.0 : Φprog(E,I,ellE,ellI)
        transmission = λT*(φ0T + φbT*ΣrI)/ΦT
        progression = λP*(φ0P + φbP*ΣrE)/ΦP
        recovery = @indicator(I > ellI, γ*I)
        waning = ω*R
        rate = transmission+progression+recovery+waning
        decay = (ψ+χ)*I + λT + λP + γ*I + ω*R - rate
        tnext = par > 0 && !isinfec(P,i,par) && tf-t > 1e-9 ? t + (tf-t)/2 : tf
        step = randexp(rng)/rate
        if t+step ≥ tnext
            ll -= decay*(tnext-t)
            t = tnext
            t < tf || break
            continue
        end
        ll -= decay*step
        t += step
        nevent += 1
        u = rand(rng)*rate
        if u < transmission+progression
            toI = u ≥ transmission
            v = toI ? (u-transmission)*ΦP/λP : u*ΦT/λT
            φ0, φb = toI ? (φ0P, φbP) : (φ0T, φbT)
            ll += log(toI ? ΦP : ΦT)
            if v ≥ φ0
                v = (v-φ0)/φb
                b, rb = zero(Name), 1.0
                for j ∈ lins
                    isinfec(P,i,tr.lins[j]) == toI && continue
                    b, rb = tr.lins[j], ratio(tr,P,i,j,e,guided)
                    v < rb && break
                    v -= rb
                end
                ll -= log(rb)
                if toI
                    infec!(P,i,b); P.ellE[i] -= 1; P.ellI[i] += 1
                else
                    expos!(P,i,b); P.ellI[i] -= 1; P.ellE[i] += 1
                end
            end
            if toI
                P.E[i] -= 1; P.I[i] += 1
            else
                P.S[i] -= 1; P.E[i] += 1
            end
        elseif u < transmission+progression+recovery
            P.I[i] -= 1; P.R[i] += 1
        else
            P.R[i] -= 1; P.S[i] += 1
        end
    end
    P.ll[i] = ll
    nevent
end

## guide ratio h(other deme)/h(own deme) of the lineage at position j
ratio(tr::Tree, P::Particles, i, j, e, guided::Bool) = begin
    guided || return 1.0
    pE, pI = guideprob(tr,j,e)
    r = isinfec(P,i,tr.lins[j]) ? pE/pI : pI/pE
    clamp(r, 1e-12, 1e12)
end

## FastSEIR.pfilter with the weighted-rate (or, with `twist`, twisted) regular part
pfilter_w(
    tr::Tree;
    Np::Integer = 1,
    β = 4.0, σ = 1.0, γ = 1.0, ω = 1.0, ψ = 0.02, χ = 0.0,
    pop = 100, S0 = 0.9, E0 = 0.0, I0 = 0.02, R0 = 0.08,
    guided::Bool = true, twist::Bool = false, tilt::Bool = true,
    trigger::Real = 1, threaded::Bool = true,
    rng::AbstractRNG = default_rng(),
) = begin
    useguide = guided && tr.guided
    p = map(Float64, (;β,σ,γ,ω,ψ,χ,pop))
    M = length(tr.time)
    P = Particles(Np,tr.nwords); Q = Particles(Np,tr.nwords)
    S, E, I, R = barycentric(Int64,(S0,E0,I0,R0),pop)
    P.S .= S; P.E .= E; P.I .= I; P.R .= R
    W = fill(1/Np,Np)
    ess = fill(Float64(Np),M); cll = zeros(M); zerofrac = zeros(M)
    ancestors = Vector{Int}(undef,Np)
    seed = rand(rng,UInt64)
    loglik = 0.0
    nevents = 0
    k0 = findlast(==(tr.time[1]),tr.time)
    for j ∈ 1:k0
        foreach_particle(P,seed,-j,threaded) do P, i, rng
            singular_part!(tr,P,i,j,p,rng;guided=useguide)
            0
        end
    end
    for k ∈ k0:M
        nevents += foreach_particle(P,seed,k,threaded) do P, i, rng
            (W[i] > 0 && isfinite(P.ll[i])) || return 0
            n = twist ? twisted_part!(tr,P,i,k,p,rng;guided=useguide,tilt) :
                regular_part!(tr,P,i,k,p,rng;guided=useguide)
            k < M && singular_part!(tr,P,i,k+1,p,rng;guided=useguide)
            n
        end
        llmax = maximum(P.ll)
        isfinite(llmax) || return FastPfilterd(-Inf,ess,cll,zerofrac,nevents)
        W .*= exp.(P.ll .- llmax)
        s = sum(W)
        cll[k] = llmax+log(s)
        loglik += cll[k]
        W ./= s
        ess[k] = 1/sum(abs2,W)
        zerofrac[k] = count(iszero,W)/Np
        P.ll .= 0.0
        if ess[k] ≤ trigger*Np
            systematic!(ancestors,W,rng)
            for (j,a) ∈ enumerate(ancestors)
                copyparticle!(Q,j,P,a)
            end
            P, Q = Q, P
            W .= 1/Np
        end
    end
    FastPfilterd(loglik,ess,cll,zerofrac,nevents)
end

end

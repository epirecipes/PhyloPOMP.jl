## Weighted alive SMC; population and colour kernels are unchanged.
module AliveFilter

using Random: AbstractRNG, default_rng
using PhyloPOMP
const F = PhyloPOMP.FastSEIR

struct TrialLimit <: Exception
    interval::Int
    trials::Int
    successes::Int
    required::Int
    total_trials::Int
    total_proposed::Int
    nevents::Int
end

Base.showerror(io::IO, e::TrialLimit) = print(io,
    "alive filter trial limit at interval ",e.interval,": ",
    e.successes,"/",e.required," survivors after ",e.trials," trials")

"""
    pfilter_alive(tr; Np=1, alive=true, batch_size=Np+1,
                  max_trials=100*(Np+1), model_params..., threaded=true, rng)

Draw independent ancestors from the previous weighted population, then use
FastSEIR's exact propagation kernels. In alive mode stop at Np+1 finite-weight
trials, discard the last survivor, and use sum(retained weights)/(T-1) as the
incremental normalizer. Positive weights need not be equal. With alive=false,
use Np ordinary trials, including deaths, as a multinomial bootstrap control.

Batches may compute trials beyond the stopping point. These are ignored by
the estimator but included in `proposed` and the population-event work count.
The cap raises TrialLimit, never a zero or partial likelihood estimate. Returns
`(pf, trials, proposed)`; `pf` has the usual FastSEIR likelihood accessors.
"""
pfilter_alive(
    tr::F.Tree; Np::Integer=1, alive::Bool=true,
    batch_size::Integer=Np+1, max_trials::Integer=100*(Np+1),
    β=4.0, σ=1.0, γ=1.0, ω=1.0, ψ=0.02, χ=0.0,
    pop=100, S0=0.9, E0=0.0, I0=0.02, R0=0.08,
    guided::Bool=true, threaded::Bool=true,
    rng::AbstractRNG=default_rng(),
) = begin
    @assert Np > 0 && batch_size > 0
    required = alive ? Np+1 : Np
    @assert max_trials ≥ required
    useguide = guided && tr.guided
    p = map(Float64,(;β,σ,γ,ω,ψ,χ,pop))
    initial = barycentric(Int64,(S0,E0,I0,R0),pop)
    M = length(tr.time)
    P = F.Particles(Np,tr.nwords)
    Q = F.Particles(Np,tr.nwords)
    U = F.Particles(batch_size,tr.nwords)
    weights = fill(1/Np,Np)
    lw = zeros(Np)
    ess = fill(Float64(Np),M)
    cll = zeros(M)
    zerofrac = zeros(M)
    trials = zeros(Int,M)
    proposed = zeros(Int,M)
    nevents = 0
    loglik = 0.0
    seed = rand(rng,UInt64)
    k0 = findlast(==(tr.time[1]),tr.time)
    for k ∈ k0:M
        cumulative = cumsum(weights)
        cumulative ./= cumulative[end]
        accepted = 0
        batch = 0
        while accepted < required
            available = max_trials-trials[k]
            available > 0 || throw(TrialLimit(k,trials[k],accepted,required,
                                             sum(trials),sum(proposed),nevents))
            ## Shrink refill batches, but amortize the 64-chunk propagation.
            wanted = alive ? max(required-accepted,64) : required-accepted
            n = min(batch_size,available,wanted)
            batch += 1
            proposed[k] += n
            nevents += F.foreach_particle(U,seed,(k,batch),threaded) do U, i, rng
                i ≤ n || return 0
                if k == k0
                    U.S[i], U.E[i], U.I[i], U.R[i] = initial
                    U.ellE[i] = U.ellI[i] = 0
                    U.col[:,i] .= 0
                    U.ll[i] = 0.0
                    for j ∈ 1:k0
                        F.singular_part!(tr,U,i,j,p,rng; guided=useguide)
                    end
                else
                    a = searchsortedlast(cumulative,rand(rng))+1
                    F.copyparticle!(U,i,P,a)
                end
                isfinite(U.ll[i]) || return 0
                events = F.regular_part!(tr,U,i,k,p,rng; proposal=:local,guided=useguide)
                k < M && F.singular_part!(tr,U,i,k+1,p,rng; guided=useguide)
                events
            end
            for i ∈ 1:n
                trials[k] += 1
                @assert isfinite(U.ll[i]) || U.ll[i] == -Inf "invalid trial weight"
                if !alive || isfinite(U.ll[i])
                    accepted += 1
                    if accepted ≤ Np
                        lw[accepted] = U.ll[i]
                        F.copyparticle!(Q,accepted,U,i)
                    end
                    accepted == required && break
                end
            end
        end
        mx = maximum(lw)
        if !isfinite(mx)
            pf = F.FastPfilterd(-Inf,ess,cll,zerofrac,nevents)
            return (;pf,trials,proposed)
        end
        weights .= exp.(lw .- mx)
        total = sum(weights)
        denominator = alive ? trials[k]-1 : Np
        cll[k] = mx+log(total/denominator)
        loglik += cll[k]
        weights ./= total
        ess[k] = 1/sum(abs2,weights)
        zerofrac[k] = count(iszero,weights)/Np
        P, Q = Q, P
    end
    pf = F.FastPfilterd(loglik,ess,cll,zerofrac,nevents)
    (;pf,trials,proposed)
end

end

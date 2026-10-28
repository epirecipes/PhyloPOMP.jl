"""
    SimSEIR

Forward simulation of the SEIRS model with non-destructive (`ψ`) and
destructive (`χ`) sampling, tracking the transmission genealogy of every
infection.  [`simulate`](@ref PhyloPOMP.SimSEIR.simulate) returns the
pruned, obscured genealogy of the samples as a PhyloPOMP
[`Genealogy`](@ref), with the same parameter conventions as the
`filter_pomp` constructors of the SEIR filter modules.
"""
module SimSEIR

using Random: AbstractRNG, default_rng
using ..PhyloPOMP
using ..PhyloPOMP: Root, Node, Sample, Name, Time, GenealNode, Genealogy
import PartiallyObservedMarkovProcesses as POMP

const ROOT = 0x00
const BRANCH = 0x01
const SAMPLE = 0x02
const END = 0x03

## remove the i-th entry of v in O(1) (order not preserved)
swappop!(v, i) = begin
    x = v[i]
    v[i] = v[end]
    pop!(v)
    x
end

"""
    simulate(; β = 4.0, σ = 1.0, γ = 1.0, ω = 1.0, ψ = 0.02, χ = 0.0,
             pop = 100, S0 = 0.9, E0 = 0.0, I0 = 0.02, R0 = 0.08,
             t0 = 0.0, time, rng = default_rng(), maxevents = 10^8)

Simulate the SEIRS genealogy process on `[t0, time]`.  Returns a named
tuple `(genealogy, nsample, state, nevents)` where `genealogy` is a
`Genealogy{PhyloPOMP.Unstructured}`, `nsample` the number of samples,
`state` the final `(S,E,I,R)` and `nevents` the number of population events.
"""
simulate(;
    β = 4.0, σ = 1.0, γ = 1.0, ω = 1.0, ψ = 0.02, χ = 0.0,
    pop = 100, S0 = 0.9, E0 = 0.0, I0 = 0.02, R0 = 0.08,
    t0 = 0.0, time::Real, rng::AbstractRNG = default_rng(),
    maxevents::Integer = 10^8,
) = begin
    S, E, I, R = POMP.barycentric(Int64, (S0, E0, I0, R0), pop)
    ## genealogy of all infections: node arrays
    nt = Float64[]; nkind = UInt8[]; npar = Int[]
    newnode!(t, kind, par) = (push!(nt, t); push!(nkind, kind); push!(npar, par); length(nt))
    cur = Int[]                     # current lineage node of each infected individual
    Eids = Int[]; Iids = Int[]      # individuals in E and in I
    for _ ∈ 1:E
        push!(cur, newnode!(t0, ROOT, 0)); push!(Eids, length(cur))
    end
    for _ ∈ 1:I
        push!(cur, newnode!(t0, ROOT, 0)); push!(Iids, length(cur))
    end
    t = Float64(t0)
    nev = 0
    nsample = 0
    while nev < maxevents
        r1 = β*S*I/pop; r2 = σ*E; r3 = γ*I; r4 = ω*R; r5 = ψ*I; r6 = χ*I
        tot = r1+r2+r3+r4+r5+r6
        tot > 0 || break
        t += randexp(rng)/tot
        t < time || break
        u = rand(rng)*tot
        nev += 1
        if (u -= r1) < 0                 # transmission
            i = Iids[rand(rng, 1:length(Iids))]
            b = newnode!(t, BRANCH, cur[i])
            cur[i] = b
            push!(cur, b); push!(Eids, length(cur))
            S -= 1; E += 1
        elseif (u -= r2) < 0             # progression
            push!(Iids, swappop!(Eids, rand(rng, 1:length(Eids))))
            E -= 1; I += 1
        elseif (u -= r3) < 0             # recovery
            i = swappop!(Iids, rand(rng, 1:length(Iids)))
            newnode!(t, END, cur[i])
            I -= 1; R += 1
        elseif (u -= r4) < 0             # waning
            R -= 1; S += 1
        elseif (u -= r5) < 0             # non-destructive sample
            i = Iids[rand(rng, 1:length(Iids))]
            cur[i] = newnode!(t, SAMPLE, cur[i])
            nsample += 1
        else                             # destructive sample
            i = swappop!(Iids, rand(rng, 1:length(Iids)))
            newnode!(t, SAMPLE, cur[i])
            I -= 1
            nsample += 1
        end
    end
    (
        genealogy = prune(nt, nkind, npar, t0, time),
        nsample = nsample,
        state = (; S, E, I, R),
        nevents = nev,
    )
end

randexp(rng) = -log(rand(rng))

## prune the full genealogy to the ancestry of the samples
prune(nt, nkind, npar, t0, time) = begin
    n = length(nt)
    marked = falses(n)
    nkept = zeros(Int, n)
    for k ∈ n:-1:1
        nkind[k] == SAMPLE && (marked[k] = true)
        if marked[k] && npar[k] > 0
            marked[npar[k]] = true
            nkept[npar[k]] += 1
        end
    end
    retained(k) = marked[k] && (nkind[k] != BRANCH || nkept[k] == 2)
    up = zeros(Int, n)              # nearest retained ancestor-or-self
    gname = zeros(Int, n)
    D = PhyloPOMP.Unstructured
    G = Genealogy{D}(t0, time)
    for k ∈ 1:n
        marked[k] || continue
        up[k] = retained(k) ? k : up[npar[k]]
        if retained(k)
            type = nkind[k] == ROOT ? Root : nkind[k] == SAMPLE ? Sample : Node
            par = npar[k] > 0 ? Name(gname[up[npar[k]]]) : nothing
            node = GenealNode{D.DemeSet}(length(G.nodes)+1, nt[k], missing, type, par)
            push!(G.nodes, node)
            gname[k] = length(G.nodes)
            isnothing(par) || push!(G.nodes[par].children, Name(gname[k]))
        end
    end
    PhyloPOMP.repair!(G)
    G
end

end

"""
    SimSEIR

Simulation of genealogies under the SEIRS model.  The population process
is simulated individual by individual, with non-destructive (rate `ψ`)
and destructive (rate `χ`) sampling of infectious individuals, while the
transmission genealogy of all infections is recorded.  The genealogy of
the samples is then extracted (pruned and obscured), so that the result
can be passed to the filters of the other SEIR modules.
"""
module SimSEIR

using Random: AbstractRNG, default_rng, randexp
using ..PhyloPOMP
using ..PhyloPOMP: Root, Node, Sample, Name, Time, GenealNode

## kinds of node in the complete transmission genealogy
const FOUNDER = 0x00        # an individual infected at the initial time
const TRANSMISSION = 0x01   # the infector's lineage splits
const SAMPLING = 0x02       # a sample is taken
const RECOVERY = 0x03       # the lineage ends unsampled

"""
    simulate(; β = 4.0, σ = 1.0, γ = 1.0, ω = 1.0, ψ = 0.02, χ = 0.0,
             pop = 100, S0 = 0.9, E0 = 0.0, I0 = 0.02, R0 = 0.08,
             t0 = 0.0, time, rng = default_rng())

Simulate the SEIRS model on the interval `[t0, time]`.  The parameters
have the same meanings and defaults as in the `filter_pomp` constructors
of the SEIR filter modules.  Returns a named tuple with the
[`Genealogy`](@ref) of the samples (`genealogy`), the number of samples
(`nsample`), the final state (`state`), and the number of population
events (`nevents`).
"""
simulate(;
    β = 4.0, σ = 1.0, γ = 1.0, ω = 1.0, ψ = 0.02, χ = 0.0,
    pop = 100, S0 = 0.9, E0 = 0.0, I0 = 0.02, R0 = 0.08,
    t0 = 0.0, time::Real, rng::AbstractRNG = default_rng(),
) = begin
    S, E, I, R = barycentric(Int64,(S0,E0,I0,R0),pop)
    ## the complete genealogy: time, kind, and parent of each node
    nodetime = Time[]; kind = UInt8[]; parent = Int[]
    newnode!(t, k, par) = begin
        push!(nodetime,t); push!(kind,k); push!(parent,par)
        length(kind)
    end
    ## the current node on the lineage of each infected individual,
    ## and the individuals in each infected compartment
    lineage = Int[]; exposed = Int[]; infectious = Int[]
    for _ ∈ 1:E
        push!(lineage, newnode!(t0,FOUNDER,0))
        push!(exposed, length(lineage))
    end
    for _ ∈ 1:I
        push!(lineage, newnode!(t0,FOUNDER,0))
        push!(infectious, length(lineage))
    end
    t = Time(t0)
    nevents = 0
    nsample = 0
    while true
        rates = (β*S*I/pop, σ*E, γ*I, ω*R, ψ*I, χ*I)
        total = sum(rates)
        total > 0 || break
        t += randexp(rng)/total
        t < time || break
        nevents += 1
        u = rand(rng)*total
        if (u -= rates[1]) < 0          # transmission
            j = rand(rng, infectious)
            lineage[j] = newnode!(t,TRANSMISSION,lineage[j])
            push!(lineage, lineage[j])
            push!(exposed, length(lineage))
            S -= 1; E += 1
        elseif (u -= rates[2]) < 0      # progression
            push!(infectious, takerandom!(rng,exposed))
            E -= 1; I += 1
        elseif (u -= rates[3]) < 0      # recovery
            j = takerandom!(rng,infectious)
            newnode!(t,RECOVERY,lineage[j])
            I -= 1; R += 1
        elseif (u -= rates[4]) < 0      # waning
            R -= 1; S += 1
        elseif (u -= rates[5]) < 0      # non-destructive sample
            j = rand(rng, infectious)
            lineage[j] = newnode!(t,SAMPLING,lineage[j])
            nsample += 1
        else                            # destructive sample
            j = takerandom!(rng,infectious)
            newnode!(t,SAMPLING,lineage[j])
            I -= 1
            nsample += 1
        end
    end
    (
        genealogy = prune(nodetime, kind, parent, t0, time),
        nsample = nsample,
        state = (;S, E, I, R),
        nevents = nevents,
    )
end

## remove and return a random element of `v` (order is not preserved)
takerandom!(rng, v) = begin
    k = rand(rng, eachindex(v))
    x = v[k]
    v[k] = v[end]
    pop!(v)
    x
end

## Prune the complete genealogy to the ancestry of the samples and convert
## it to a Genealogy: founders and transmissions without sampled
## descendants disappear, and a transmission with sampled descendants along
## only one of its two lineages is not visible.
prune(nodetime, kind, parent, t0, time) = begin
    n = length(kind)
    sampled = falses(n)         # is the node, or one of its descendants, a sample?
    nsampled = zeros(Int,n)     # number of children with sampled descendants
    for k ∈ Iterators.reverse(1:n)
        if kind[k] == SAMPLING
            sampled[k] = true
        end
        if sampled[k] && parent[k] > 0
            sampled[parent[k]] = true
            nsampled[parent[k]] += 1
        end
    end
    visible(k) = sampled[k] && (kind[k] ≠ TRANSMISSION || nsampled[k] == 2)
    D = PhyloPOMP.Unstructured
    G = Genealogy{D}(t0,time)
    name = zeros(Int,n)         # name of the nearest visible ancestor (or self)
    for k ∈ 1:n
        sampled[k] || continue
        if visible(k)
            type = kind[k] == FOUNDER ? Root : kind[k] == SAMPLING ? Sample : Node
            par = parent[k] > 0 ? Name(name[parent[k]]) : nothing
            push!(G.nodes, GenealNode{D.DemeSet}(length(G.nodes)+1,nodetime[k],missing,type,par))
            isnothing(par) || push!(G.nodes[par].children, Name(length(G.nodes)))
            name[k] = length(G.nodes)
        else
            name[k] = name[parent[k]]
        end
    end
    PhyloPOMP.repair!(G)
    G
end

end

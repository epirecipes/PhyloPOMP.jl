"""
    ExactSEIR

Exact computation of the likelihood of a genealogy under the SEIRS model
for *small* populations.  The filter equation of King, Lin & Ionides is
solved deterministically on the joint space of population states
`(S,E,I,R)` and lineage colorings, by uniformization between
genealogical events.  Since no importance sampling is involved, the
result can serve as a reference for checking the particle filters (e.g.,
[`GuidedSEIR`](@ref PhyloPOMP.GuidedSEIR)).

The size of the state space is (number of population states) ×
2^(number of lineages present), so only small examples are feasible.
"""
module ExactSEIR

using ..PhyloPOMP
using ..PhyloPOMP: Root, Node, Sample, Name, Time

## A state is a population state together with a coloring, represented as
## a bitmask: bit b-1 is set if and only if lineage b is in deme I.
const State = Tuple{NTuple{4,Int},UInt64}

bit(b::Integer) = one(UInt64) << (b-1)

isinfec(m::UInt64, b::Integer) = m & bit(b) ≠ 0

"""
    Space(lins, totals)

The feasible states when the lineages `lins` are present and the total
population size is one of `totals`.
"""
struct Space
    lins::Vector{Name}
    states::Vector{State}
    index::Dict{State,Int}
    Space(lins::Vector{Name}, totals) = begin
        states = State[]
        cols = colorings(lins)
        for N ∈ totals, S ∈ 0:N, E ∈ 0:(N-S), I ∈ 0:(N-S-E), m ∈ cols
            R = N-S-E-I
            ellI = count_ones(m)
            ellE = length(lins)-ellI
            if E ≥ ellE && I ≥ ellI
                push!(states, ((S,E,I,R), m))
            end
        end
        new(lins, states, Dict(s => k for (k,s) ∈ enumerate(states)))
    end
end

## all colorings of the lineages `lins`
colorings(lins) = map(0:(2^length(lins)-1)) do k
    m = zero(UInt64)
    for (j,b) ∈ enumerate(lins)
        if (k >> (j-1)) & 1 == 1
            m |= bit(b)
        end
    end
    m
end

"""
    Generator(sp, params)

The generator of the regular part of the filter equation on the state
space `sp`: `dw/dt = Q w`, where `Q` has off-diagonal entries given by
`(from, to, rate)` triples and diagonal `diag`.
"""
struct Generator
    from::Vector{Int}
    to::Vector{Int}
    rate::Vector{Float64}
    diag::Vector{Float64}
    Generator(sp::Space, p) = begin
        (; β, σ, γ, ω, ψ, χ, pop) = p
        from = Int[]; to = Int[]; rate = Float64[]
        diag = Vector{Float64}(undef,length(sp.states))
        for (j, ((S,E,I,R), m)) ∈ enumerate(sp.states)
            ellI = count_ones(m)
            ellE = length(sp.lins)-ellI
            diag[j] = -(β*S*I/pop + σ*E + γ*I + ω*R + (ψ+χ)*I)
            add!(x, m, r) = if r > 0
                push!(from, j); push!(to, sp.index[(x, m)]); push!(rate, r)
            end
            if S > 0 && I > 0               # transmission
                a = β*S*I/pop
                add!((S-1,E+1,I,R), m, a*(E+1-ellE)/(E+1))
                for b ∈ sp.lins
                    if isinfec(m,b)         # lineage b passes to the infectee
                        add!((S-1,E+1,I,R), m & ~bit(b), a*(I-ellI+1)/((E+1)*I))
                    end
                end
            end
            if E > 0                        # progression
                a = σ*E
                if E > ellE
                    add!((S,E-1,I+1,R), m, a*(I+1-ellI)/(I+1))
                end
                for b ∈ sp.lins
                    if !isinfec(m,b)        # lineage b progresses
                        add!((S,E-1,I+1,R), m | bit(b), a/(I+1))
                    end
                end
            end
            if I > ellI                     # recovery
                add!((S,E,I-1,R+1), m, γ*I)
            end
            if R > 0                        # waning
                add!((S+1,E,I,R-1), m, ω*R)
            end
        end
        new(from, to, rate, diag)
    end
end

"""
    propagate!(w, Q, dt)

Advance `w` by `dt` under generator `Q` by uniformization.  `w` is
normalized on return; the log of its total mass is returned.
"""
propagate!(
    w::Vector{Float64},
    Q::Generator,
    dt::Real,
) = begin
    Λ = maximum(abs, Q.diag; init=0.0)
    (dt > 0 && Λ > 0) || return 0.0
    ## with M = I + Q/Λ, exp(Q h) = Σₖ Pois(k; Λh) Mᵏ, all terms nonnegative
    nstep = ceil(Int, Λ*dt/16)
    λh = Λ*dt/nstep
    term = similar(w); next = similar(w); acc = similar(w)
    logmass = 0.0
    for _ ∈ 1:nstep
        term .= w
        pk = exp(-λh)
        acc .= pk .* term
        k = 0
        while k ≤ λh || pk*sum(term) > 1e-17*sum(acc)
            k += 1
            @assert k < 100_000 "uniformization failed to converge"
            next .= term .* (1 .+ Q.diag ./ Λ)
            for (f, t, r) ∈ zip(Q.from, Q.to, Q.rate)
                next[t] += r/Λ*term[f]
            end
            term, next = next, term
            pk *= λh/k
            acc .+= pk .* term
        end
        s = sum(acc)
        s > 0 || return -Inf
        w .= acc ./ s
        logmass += log(s)
    end
    logmass
end

## Apply the singular part of the filter equation for a genealogical node
## (with parent lineage `parlin` and child lineages `chillins`), mapping the
## weights `w` on `sp` to weights on `sp2`.
singular(
    w::Vector{Float64},
    sp::Space,
    sp2::Space,
    type,
    parlin::Name,
    chillins::Vector{Name},
    p,
) = begin
    (; β, ψ, χ, pop) = p
    w2 = zeros(length(sp2.states))
    deposit!(x, m, v) = begin
        k = get(sp2.index, (x, m), 0)   # zero if the state is infeasible
        k > 0 && (w2[k] += v)
    end
    for (j, ((S,E,I,R), m)) ∈ enumerate(sp.states)
        v = w[j]
        v > 0 || continue
        if type == Root                 # the new lineage is in either deme
            c = chillins[1]
            deposit!((S,E,I,R), m, v)
            deposit!((S,E,I,R), m | bit(c), v)
        elseif isinfec(m,parlin)        # otherwise, the parent must be infectious
            m0 = m & ~bit(parlin)
            if type == Sample && isempty(chillins)
                deposit!((S,E,I,R), m0, v*ψ*(I-count_ones(m0)))
                deposit!((S,E,I-1,R), m0, v*χ*I)
            elseif type == Sample
                deposit!((S,E,I,R), m0 | bit(chillins[1]), v*ψ)
            elseif type == Node && S > 0
                c1, c2 = chillins
                wt = v*β*S/(pop*(E+1))
                deposit!((S-1,E+1,I,R), m0 | bit(c1), wt)
                deposit!((S-1,E+1,I,R), m0 | bit(c2), wt)
            end
        end
    end
    w2
end

## the state space and generator for the lineages `lins`, memoized in `cache`
space!(cache, lins, totals, p) = get!(cache, lins) do
    sp = Space(lins, totals)
    sp, Generator(sp, p)
end

"""
    loglik(g; β = 4.0, σ = 1.0, γ = 1.0, ω = 1.0, ψ = 0.02, χ = 0.0,
           pop = 100, S0 = 0.9, E0 = 0.0, I0 = 0.02, R0 = 0.08)

Exact log likelihood of the [`Genealogy`](@ref) `g` under the SEIRS
model.  The parameters have the same meanings and defaults as in the
`filter_pomp` constructors of the other SEIR modules.
"""
loglik(
    g::Genealogy;
    β = 4.0, σ = 1.0, γ = 1.0, ω = 1.0, ψ = 0.02, χ = 0.0,
    pop = 100, S0 = 0.9, E0 = 0.0, I0 = 0.02, R0 = 0.08,
) = begin
    @assert g.nsample ≤ 64 "too many samples for the exact computation"
    p = map(Float64, (;β,σ,γ,ω,ψ,χ,pop))
    x0 = Tuple(barycentric(Int64,(S0,E0,I0,R0),pop))
    totals = χ > 0 ? (0:sum(x0)) : (sum(x0):sum(x0))
    cache = Dict{Vector{Name},Tuple{Space,Generator}}()
    lins = Name[]
    sp, Q = space!(cache, lins, totals, p)
    w = zeros(length(sp.states))
    w[sp.index[(x0, zero(UInt64))]] = 1
    ll = 0.0
    t = timezero(g)
    for n ∈ eachindex(g)
        ll += propagate!(w, Q, g[n].slate-t)
        t = g[n].slate
        parlin = g[n].lineage
        chillins = Name[g[c].lineage for c ∈ g[n].children]
        lins = sort!(union(setdiff(lins,[parlin]), chillins))
        sp2, Q = space!(cache, lins, totals, p)
        w = singular(w, sp, sp2, g[n].type, parlin, chillins, p)
        sp = sp2
        s = sum(w)
        s > 0 || return -Inf
        w ./= s
        ll += log(s)
    end
    ll + propagate!(w, Q, g.time-t)
end

"""
    events(g)

The genealogical events of `g` as a vector of named tuples
`(t, type, parent, children)`, e.g., for exporting test cases.
"""
events(g::Genealogy) = map(eachindex(g)) do n
    (
        t = g[n].slate,
        type = lowercase(String(Symbol(g[n].type))),
        parent = Int(g[n].lineage),
        children = Int[g[c].lineage for c ∈ g[n].children],
    )
end

end

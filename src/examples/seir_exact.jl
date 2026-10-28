"""
    ExactSEIR

Deterministic (non-Monte Carlo) solution of the SEIRS filter equation of
King, Lin & Ionides (2024) for *small* populations.  It enumerates the
joint space of population states `(S,E,I,R)` and lineage colorings, and
integrates the filter equation exactly (by uniformization) between
genealogical events.  It is independent of the particle-filter code and
of any importance-sampling proposal, so it serves as a reference
likelihood for validating the particle filters ([`NaiveSEIR`](@ref
PhyloPOMP.NaiveSEIR), [`GuidedSEIR`](@ref PhyloPOMP.GuidedSEIR), ...).

The cost grows like (number of population states) × 2^(number of
lineages), so it is only practical for tiny examples.
"""
module ExactSEIR

using ..PhyloPOMP
using ..PhyloPOMP: Root, Node, Sample, Name, Time, Genealogy
import PartiallyObservedMarkovProcesses as POMP

const PopState = NTuple{4,Int}
const Key = Tuple{PopState,UInt64}

## joint state space for a given set of lineages; mask bit k set ⇔ lins[k] in deme I
struct Space
    lins::Vector{Name}
    states::Vector{Key}
    index::Dict{Key,Int}
end

Space(lins::Vector{Name}, totals) = begin
    A = length(lins)
    @assert A ≤ 20 "too many lineages for the exact solver"
    states = Key[]
    for N ∈ totals, S ∈ 0:N, E ∈ 0:(N-S), I ∈ 0:(N-S-E)
        R = N-S-E-I
        for m ∈ UInt64(0):UInt64(2^A-1)
            lI = count_ones(m)
            if E ≥ A-lI && I ≥ lI
                push!(states, ((S,E,I,R), m))
            end
        end
    end
    Space(lins, states, Dict(s => k for (k,s) ∈ enumerate(states)))
end

## sparse weighted generator: dw/dt = G w with G = offdiag + Diagonal(diag)
struct Generator
    src::Vector{Int}
    dst::Vector{Int}
    rate::Vector{Float64}
    diag::Vector{Float64}
end

inI(m::UInt64, k::Int) = (m >> (k-1)) & one(UInt64) == one(UInt64)
setI(m::UInt64, k::Int) = m | (one(UInt64) << (k-1))
setE(m::UInt64, k::Int) = m & ~(one(UInt64) << (k-1))

Generator(sp::Space, p) = begin
    (; β, σ, γ, ω, ψ, χ, pop) = p
    A = length(sp.lins)
    src = Int[]; dst = Int[]; rate = Float64[]
    diag = zeros(Float64, length(sp.states))
    add!(j, key, r) = if r > 0
        push!(src, j); push!(dst, sp.index[key]); push!(rate, r)
    end
    for (j, ((S,E,I,R), m)) ∈ enumerate(sp.states)
        lI = count_ones(m)
        lE = A-lI
        diag[j] = -(β*S*I/pop + σ*E + γ*I + ω*R + (ψ+χ)*I)
        if S ≥ 1 && I ≥ 1               # transmission
            a = β*S*I/pop
            E2 = E+1
            add!(j, ((S-1,E2,I,R), m), a*(E2-lE)/E2)
            for k ∈ 1:A
                if inI(m,k)              # lineage k moves I → E
                    add!(j, ((S-1,E2,I,R), setE(m,k)), a*(I-(lI-1))/(E2*I))
                end
            end
        end
        if E ≥ 1                         # progression
            a = σ*E
            I2 = I+1
            if E-1 ≥ lE
                add!(j, ((S,E-1,I2,R), m), a*(I2-lI)/I2)
            end
            for k ∈ 1:A
                if !inI(m,k)             # lineage k moves E → I
                    add!(j, ((S,E-1,I2,R), setI(m,k)), a/I2)
                end
            end
        end
        if I ≥ 1 && I-1 ≥ lI             # recovery of a non-lineage-carrier
            add!(j, ((S,E,I-1,R+1), m), γ*I)
        end
        if R ≥ 1                         # waning
            add!(j, ((S+1,E,I,R-1), m), ω*R)
        end
    end
    Generator(src, dst, rate, diag)
end

## y = M x with M = I + G/Λ (all entries of M nonnegative)
uniform_step!(y, x, G::Generator, Λ) = begin
    @inbounds for j ∈ eachindex(x)
        y[j] = x[j]*(1+G.diag[j]/Λ)
    end
    @inbounds for k ∈ eachindex(G.src)
        y[G.dst[k]] += G.rate[k]/Λ*x[G.src[k]]
    end
    y
end

## advance w by dt, renormalizing; returns the log of the mass change
propagate!(w::Vector{Float64}, G::Generator, dt::Real) = begin
    dt > 0 || return 0.0
    Λ = maximum(abs, G.diag; init = 0.0)
    Λ > 0 || return 0.0
    nsub = max(1, ceil(Int, Λ*dt/16))
    h = dt/nsub
    λh = Λ*h
    logscale = 0.0
    term = similar(w)
    tmp = similar(w)
    acc = similar(w)
    for _ ∈ 1:nsub
        copyto!(term, w)
        pk = exp(-λh)
        acc .= pk .* term
        k = 0
        while true
            k += 1
            uniform_step!(tmp, term, G, Λ)
            term, tmp = tmp, term
            pk *= λh/k
            contrib = pk*sum(term)
            acc .+= pk .* term
            (k > λh && contrib ≤ 1e-17*sum(acc)) && break
            k > 100_000 && error("uniformization failed to converge")
        end
        s = sum(acc)
        s > 0 || return -Inf
        w .= acc ./ s
        logscale += log(s)
    end
    logscale
end

## apply the singular operator of genealogical node `n`, mapping w from `sp` to `sp2`
singular(w, sp::Space, sp2::Space, type, parlin, chillins, p) = begin
    (; β, ψ, χ, pop) = p
    w2 = zeros(Float64, length(sp2.states))
    pos(sp, lin) = findfirst(==(lin), sp.lins)
    put!(x, m, v) = begin
        key = (x, m)
        k = get(sp2.index, key, 0)
        k > 0 && v > 0 && (w2[k] += v)
    end
    ## translate a mask on sp.lins (minus `drop`) to a mask on sp2.lins
    remap(m, drop) = begin
        m2 = zero(UInt64)
        for (k, lin) ∈ enumerate(sp.lins)
            if lin ≠ drop && inI(m,k)
                m2 = setI(m2, pos(sp2,lin))
            end
        end
        m2
    end
    for (j, ((S,E,I,R), m)) ∈ enumerate(sp.states)
        v = w[j]
        v > 0 || continue
        if type == Root
            c = chillins[1]
            m2 = remap(m, zero(Name))
            put!((S,E,I,R), m2, v)                     # lineage c in E (bit unset)
            put!((S,E,I,R), setI(m2, pos(sp2,c)), v)   # lineage c in I
        else
            kp = pos(sp, parlin)
            inI(m, kp) || continue                     # parent must be infectious
            m2 = remap(m, parlin)
            if type == Sample && isempty(chillins)
                lI2 = count_ones(m2)
                put!((S,E,I,R), m2, v*ψ*(I-lI2))
                χ > 0 && put!((S,E,I-1,R), m2, v*χ*I)
            elseif type == Sample
                put!((S,E,I,R), setI(m2, pos(sp2,chillins[1])), v*ψ)
            elseif type == Node
                S ≥ 1 || continue
                wt = v*β*S/(pop*(E+1))
                c1, c2 = chillins
                put!((S-1,E+1,I,R), setI(m2, pos(sp2,c1)), wt)
                put!((S-1,E+1,I,R), setI(m2, pos(sp2,c2)), wt)
            end
        end
    end
    w2
end

"""
    loglik(g; β = 4.0, σ = 1.0, γ = 1.0, ω = 1.0, ψ = 0.02, χ = 0.0,
           pop = 100, S0 = 0.9, E0 = 0.0, I0 = 0.02, R0 = 0.08)

Exact log likelihood of the [`Genealogy`](@ref) `g` under the SEIRS
model.  Parameters and initial-state conventions are identical to
those of the `filter_pomp` constructors of the SEIR filter modules.
"""
loglik(
    g::Genealogy;
    β = 4.0, σ = 1.0, γ = 1.0, ω = 1.0, ψ = 0.02, χ = 0.0,
    pop = 100, S0 = 0.9, E0 = 0.0, I0 = 0.02, R0 = 0.08,
) = begin
    p = map(Float64, (; β, σ, γ, ω, ψ, χ, pop))
    x0 = Tuple(POMP.barycentric(Int64, (S0, E0, I0, R0), pop))::PopState
    N0 = sum(x0)
    totals = χ > 0 ? (0:N0) : (N0:N0)
    sp = Space(Name[], totals)
    w = zeros(Float64, length(sp.states))
    w[sp.index[(x0, zero(UInt64))]] = 1.0
    cache = Dict{Vector{Name},Tuple{Space,Generator}}()
    getspace(lins) = get!(cache, lins) do
        s = Space(lins, totals)
        (s, Generator(s, p))
    end
    ll = 0.0
    t = PhyloPOMP.timezero(g)
    lins = Name[]
    for n ∈ eachindex(g)
        node = g[n]
        sp, G = getspace(lins)
        ll += propagate!(w, G, node.slate - t)
        t = node.slate
        parlin = node.lineage
        chillins = Name[g[c].lineage for c ∈ node.children]
        lins2 = sort!(union(setdiff(lins, [parlin]), chillins))
        sp2, _ = getspace(lins2)
        w = singular(w, sp, sp2, node.type, parlin, chillins, p)
        s = sum(w)
        s > 0 || return -Inf
        w ./= s
        ll += log(s)
        lins = lins2
    end
    _, G = getspace(lins)
    ll += propagate!(w, G, g.time - t)
    ll
end

"""
    events(g)

Flatten a [`Genealogy`](@ref) into a vector of named tuples
`(t, type, parent, children)` (useful for exporting test cases).
"""
events(g::Genealogy) = map(eachindex(g)) do n
    node = g[n]
    (
        t = node.slate,
        type = lowercase(String(Symbol(node.type))),
        parent = Int(node.lineage),
        children = Int[g[c].lineage for c ∈ node.children],
    )
end

end

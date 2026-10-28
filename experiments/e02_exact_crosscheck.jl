## E02: cross-check the Julia exact solver (ExactSEIR, uniformization) against the
## independent Python/mpmath dense-expm solver on hand-made genealogies.
include("common.jl")

cases = [
  (name="one_sample", newick="([&&PhyloPOMP type=sample]:1.1);", time=2.0,
   par=(β=2.0,σ=1.5,γ=0.7,ω=0.3,ψ=0.5,χ=0.0,pop=4,S0=0.5,E0=0.0,I0=0.5,R0=0.0)),
  (name="branch", newick="(([&&PhyloPOMP type=sample]:0.6,[&&PhyloPOMP type=sample]:1.1)[&&PhyloPOMP type=node]:0.4);", time=2.0,
   par=(β=2.0,σ=1.5,γ=0.7,ω=0.3,ψ=0.5,χ=0.0,pop=4,S0=0.5,E0=0.25,I0=0.25,R0=0.0)),
  (name="inline", newick="(([&&PhyloPOMP type=sample]:0.7)[&&PhyloPOMP type=sample]:0.5);", time=1.6,
   par=(β=2.0,σ=1.5,γ=0.7,ω=0.3,ψ=0.5,χ=0.0,pop=3,S0=1/3,E0=0.0,I0=2/3,R0=0.0)),
  (name="two_roots", newick=["([&&PhyloPOMP type=sample]:0.8);","([&&PhyloPOMP type=sample]:1.3);"], time=1.5,
   par=(β=2.0,σ=1.5,γ=0.7,ω=0.3,ψ=0.5,χ=0.0,pop=4,S0=0.25,E0=0.25,I0=0.5,R0=0.0)),
  (name="destructive", newick="([&&PhyloPOMP type=sample]:1.1);", time=2.0,
   par=(β=2.0,σ=1.5,γ=0.7,ω=0.3,ψ=0.5,χ=0.4,pop=4,S0=0.5,E0=0.0,I0=0.5,R0=0.0)),
]

export_case(c, g) = begin
    p = c.par
    x0 = collect(POMP.barycentric(Int64, (p.S0, p.E0, p.I0, p.R0), p.pop))
    (name=c.name,
     params=(beta=p.β, sigma=p.σ, gamma=p.γ, omega=p.ω, psi=p.ψ, chi=p.χ, pop=p.pop),
     x0=x0, t0=timezero(g), T=g.time, events=ExactSEIR.events(g))
end

rows = NamedTuple[]
exported = String[]
for c ∈ cases
    g = parse_newick(c.newick, time=c.time)
    t = @elapsed ll = ExactSEIR.loglik(g; c.par...)
    push!(rows, (case=c.name, julia_loglik=ll, seconds=t))
    push!(exported, tojson(export_case(c, g)))
end
json_path = joinpath(RESULTS, "e02_cases.json")
write(json_path, "[" * join(exported, ",\n") * "]")
py = joinpath(LAB, "maths", "sympy", ".venv", "bin", "python")
script = joinpath(LAB, "maths", "sympy", "exact_mpmath.py")
pyout = read(`$py $script $json_path`, String)
pyvals = Dict(m.captures[1] => parse(BigFloat, m.captures[2])
              for m ∈ eachmatch(r"\"(\w+)\": \"([-\d\.e]+)\"", pyout))
out = map(rows) do r
    ref = pyvals[r.case]
    (case=r.case, julia=r.julia_loglik, mpmath=Float64(ref),
     abs_diff=abs(r.julia_loglik - Float64(ref)), seconds=r.seconds)
end
for r ∈ out
    @printf("%-12s julia=% .12f  mpmath=% .12f  |Δ|=%.2e  (%.3fs)\n",
            r.case, r.julia, r.mpmath, r.abs_diff, r.seconds)
end
write_csv(joinpath(RESULTS, "e02_exact_crosscheck.csv"), out)
@assert all(r -> r.abs_diff < 1e-9, out) "exact solvers disagree"
println("E02 PASS: Julia uniformization agrees with mpmath expm to < 1e-9")

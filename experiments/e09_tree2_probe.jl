## E09 probe: feasibility/timing on seir_trees[2] (1562 nodes, 864 samples, T≈399.45)
include("common.jl")
using PhyloPOMP: FastSEIR
g = parse_newick(GuidedSEIR.seir_trees[2])
par = (β=4.0, σ=1.0, γ=1.0, ω=1.0, ψ=0.02, χ=0.0, pop=100, S0=0.9, E0=0.0, I0=0.02, R0=0.08)
println("tree 2: nodes=$(length(g)) samples=$(g.nsample) T=$(g.time)")
## maximum number of simultaneous lineages
let A = 0, amax = 0
    for n ∈ g.nodes
        A += length(n.children) - (n.type == PhyloPOMP.Root ? 0 : 1)
        amax = max(amax, A)
    end
    println("max simultaneous lineages ≈ $amax")
end
tr = FastSEIR.Tree(g, guide(g, guide_for(GuidedSEIR), GuidedSEIR.knowledge!))
for (lab, opt) ∈ [("fast/local/post/guided (8 thr)", FastSEIR.Options()),
                  ("fast/naive/pre (8 thr)", FastSEIR.Options(proposal=:naive, guided=false, order=:pre))]
    FastSEIR.pfilter(tr, par; Np=100, opt)
    for Np ∈ (1000, 4000)
        st = @timed r = FastSEIR.pfilter(tr, par; Np, opt, diagnostics=true)
        @printf("%-30s Np=%5d logLik=%.2f time=%.2fs alloc=%.0fMiB minESS=%.1f meanZero=%.3f events/particle=%.0f\n",
                lab, Np, r.logLik, st.time, st.bytes/2^20, minimum(filter(!isnan, r.ess)),
                mean(filter(!isnan, r.zerofrac)), r.nevents/Np)
    end
end
p = make_pomp(:guided, g; par...)
pfilter(p; Np=10)
for Np ∈ (100, 400)
    st = @timed pf = pfilter(p; Np)
    @printf("pomp/guided Np=%d logLik=%.2f time=%.2fs alloc=%.0fMiB\n", Np, logLik(pf), st.time, st.bytes/2^20)
end

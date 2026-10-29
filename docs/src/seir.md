# SEIR model

## Naïve filter

```@autodocs
Modules = [PhyloPOMP.NaiveSEIR]
Order   = [:module, :function, :macro, :type, :constant]
```

An example constructing the pomp object and running a particle filter.

```@example
using Random: seed!
using PhyloPOMP
using PhyloPOMP.NaiveSEIR

seed!(351956486)

g = parse_newick(NaiveSEIR.seir_trees[1], time = 50.0)
p = NaiveSEIR.filter_pomp(g)
pf = pfilter(p, Np = 1000)
round(logLik(pf),digits=1)
```

## Soft-guided filter

```@autodocs
Modules = [PhyloPOMP.SoftSEIR]
Order   = [:module, :function, :macro, :type, :constant]
```

An example constructing the pomp object and running a particle filter.

```@example
using Random: seed!
using PhyloPOMP
using PhyloPOMP.SoftSEIR
using PhyloPOMP.SoftSEIR.Demes: Expos, Infec

seed!(351956486)

g = parse_newick(SoftSEIR.seir_trees[1], time = 50.0)
p = SoftSEIR.filter_pomp(g,fsmarkov(Expos=>0.1,Infec=>1,(Expos,Infec)=>1))
pf = pfilter(p, Np = 1000)
round(logLik(pf),digits=1)
```


## Guided filter

```@autodocs
Modules = [PhyloPOMP.GuidedSEIR]
Order   = [:module, :function, :macro, :type, :constant]
```

An example constructing the pomp object and running a particle filter.

```@example
using Random: seed!
using PhyloPOMP
using PhyloPOMP.GuidedSEIR
using PhyloPOMP.GuidedSEIR.Demes: Expos, Infec

seed!(351956486)

g = parse_newick(GuidedSEIR.seir_trees[1], time = 50.0)
p = GuidedSEIR.filter_pomp(g,fsmarkov(Expos=>0.1,Infec=>1,(Expos,Infec)=>1))
pf = pfilter(p, Np = 1000)
round(logLik(pf),digits=1)
```


## Hard-guided filter

```@autodocs
Modules = [PhyloPOMP.HardSEIR]
Order   = [:module, :function, :macro, :type, :constant]
```

An example constructing the pomp object and running a particle filter.

```@example
using Random: seed!
using PhyloPOMP
using PhyloPOMP.HardSEIR
using PhyloPOMP.HardSEIR.Demes: Expos, Infec

seed!(351956486)

g = parse_newick(HardSEIR.seir_trees[1], time = 50.0)
p = HardSEIR.filter_pomp(g,fsmarkov(Expos=>0.1,Infec=>1,(Expos,Infec)=>1))
pf = pfilter(p, Np = 1000)
round(logLik(pf),digits=1)
```

## Fast streaming filter

```@autodocs
Modules = [PhyloPOMP.FastSEIR]
Order   = [:module, :function, :macro, :type, :constant]
```

An example running the fast filter with a filter guide.

```@example
using Random: Xoshiro
using PhyloPOMP
using PhyloPOMP.FastSEIR
using PhyloPOMP.FastSEIR.Demes: Expos, Infec

g = parse_newick(FastSEIR.seir_trees[1], time = 50.0)
tr = FastSEIR.Tree(g, fsmarkov(Expos=>0.1, Infec=>1, (Expos,Infec)=>1))
pf = pfilter(tr, Np = 1000, rng = Xoshiro(351956486))
round(logLik(pf), digits = 1)
```

Parameters can be estimated by iterated filtering with `mif`, which
takes the same perturbation kernels and cooling schedules as the `mif`
method on `PompObject`s.  Here `β` and `γ` start from 8 and 0.5.

```@example
using Random: Xoshiro, seed!
using PhyloPOMP
using PhyloPOMP.FastSEIR
using PhyloPOMP.FastSEIR.Demes: Expos, Infec

g = parse_newick(FastSEIR.seir_trees[1], time = 50.0)
tr = FastSEIR.Tree(g, fsmarkov(Expos=>0.1, Infec=>1, (Expos,Infec)=>1))
seed!(1066)
mf = mif(
    tr, Np = 1000, Nmif = 20,
    perturbations = @perturbn(β ~ LogNormal(0.02), γ ~ LogNormal(0.02)),
    cooling = geometric_cooling(0.5),
    β = 8.0, γ = 0.5,
    rng = Xoshiro(1066),
)
map(x -> round(x, digits = 2), coef(mf)[(:β, :γ)]), round(logLik(mf), digits = 1)
```

# Covariate adjustment: which sets Z make P(y | do(x)) = ∑_z P(y | x, z) P(z)?
#
# The generalised adjustment criterion (Shpitser, VanderWeele & Robins 2010;
# Perković et al. 2018) is complete for DAGs and for DAGs with latent
# variables (an ADMG is read through its canonical DAG): Z is valid iff it
# contains no node of Forb(X, Y) and it blocks every non-causal path, i.e.
# X ⟂ Y | Z in the proper back-door graph. The optimal set of Henckel,
# Perković & Maathuis (2022) gives the smallest asymptotic variance among all
# valid sets.

_adjdag(g::DAG) = (g, Set(g.nodes))
_adjdag(g::ADMG) = (canonical_dag(g), Set(nodes(g)))

# Nodes on proper causal paths from X to Y, excluding X.
function _causal_nodes(G::DAG, X::Set{RV}, Y::Set{RV})
    down = Set{RV}()                     # reachable from X without re-entering X
    stack = RV[c for x in X for c in children(G, x) if !(c in X)]
    while !isempty(stack)
        v = pop!(stack)
        v in down && continue
        push!(down, v)
        append!(stack, (c for c in children(G, v) if !(c in X)))
    end
    up = Set{RV}()                       # ancestors of Y (and Y) avoiding X
    stack = RV[y for y in Y if !(y in X)]
    while !isempty(stack)
        v = pop!(stack)
        v in up && continue
        push!(up, v)
        append!(stack, (p for p in G.parents[v] if !(p in X)))
    end
    return intersect(down, up)
end

function _adjustment_setup(g, X, Y)
    G, observed = _adjdag(g)
    Xs, Ys = Set(_rvlist(X)), Set(_rvlist(Y))
    isempty(intersect(Xs, Ys)) || throw(ArgumentError("treatment and outcome must be disjoint"))
    for v in union(Xs, Ys)
        v in observed || throw(ArgumentError(string(v.name, " is not an observed node of the graph")))
    end
    cn = _causal_nodes(G, Xs, Ys)
    forb = union(cn, descendants(G, collect(cn)), Xs)
    # proper back-door graph: drop the first edge of every proper causal path
    pa = Dict{RV,Vector{RV}}(v => (v in cn ? RV[p for p in ps if !(p in Xs)] : copy(ps)) for (v, ps) in G.parents)
    pbd = DAG(G.nodes, pa)
    return (; G, observed, Xs, Ys, cn, forb, pbd)
end

"""
    isadjustment(g, X, Y, Z) -> Bool

Whether `Z` is a valid adjustment set for the effect of `X` on `Y` in the
DAG or ADMG `g`: `P(Y | do(X)) = ∑_Z P(Y | X, Z) P(Z)` for every distribution
compatible with `g`. Uses the generalised adjustment criterion, which is
complete, so `false` means that formula fails for some distribution. In an
ADMG the bidirected edges are hidden confounders, and `Z` must be observed.
"""
function isadjustment(g::Union{DAG,ADMG}, X, Y, Z)
    s = _adjustment_setup(g, X, Y)
    Zs = Set(_rvlist(Z))
    all(z -> z in s.observed, Zs) || return false
    isempty(intersect(Zs, s.forb)) || return false
    isempty(intersect(Zs, s.Ys)) || return false
    return d_separated(s.pbd, collect(s.Xs), collect(s.Ys), collect(Zs))
end

"""
    adjustment_sets(g, X, Y; minimal = true, maxcandidates = 18) -> Vector{Vector{RV}}

The valid adjustment sets for the effect of `X` on `Y` in a DAG or ADMG,
smallest first (see [`isadjustment`](@ref)). With `minimal = true` only the
minimal ones (no proper subset is valid); otherwise every valid set. Empty
when no adjustment set exists (as under the front door, where
[`identify`](@ref) still succeeds). Candidates are the observed nodes outside
`Forb(X, Y)`; their subsets are searched, so at most `maxcandidates` of them.
"""
function adjustment_sets(g::Union{DAG,ADMG}, X, Y; minimal::Bool=true, maxcandidates::Integer=18)
    s = _adjustment_setup(g, X, Y)
    cand = sort!(RV[v for v in s.observed if !(v in s.forb) && !(v in s.Ys)])
    length(cand) > maxcandidates && throw(ArgumentError(string(length(cand),
        " candidate variables: too many subsets to search (raise maxcandidates, or use optimal_adjustment)")))
    found = Vector{RV}[]
    for k in 0:length(cand), Z in _combinations(cand, k)
        minimal && any(F -> issubset(F, Z), found) && continue
        d_separated(s.pbd, collect(s.Xs), collect(s.Ys), Z) && push!(found, RV[Z...])
    end
    return found
end

"""
    optimal_adjustment(g::DAG, X, Y) -> Vector{RV}

The optimal adjustment set `O(X, Y) = pa(cn(X, Y)) ∖ Forb(X, Y)` (Henckel,
Perković & Maathuis 2022): the parents of the nodes on causal paths, minus
the forbidden nodes. Among all valid adjustment sets it gives the adjusted
estimator with the smallest asymptotic variance (linear models; also optimal
in a broad nonparametric sense). It is valid whenever any adjustment set is;
an `ArgumentError` says when none is. For DAGs without hidden variables.
"""
function optimal_adjustment(g::DAG, X, Y)
    s = _adjustment_setup(g, X, Y)
    O = sort!(RV[p for c in s.cn for p in s.G.parents[c] if !(p in s.forb)] |> unique)
    isadjustment(g, X, Y, O) || throw(ArgumentError("no valid adjustment set exists for this effect"))
    return O
end
optimal_adjustment(g::ADMG, X, Y) = isempty(g.bidirected) ? optimal_adjustment(g.dag, X, Y) :
    throw(ArgumentError("the optimal adjustment set is defined for DAGs without hidden variables"))

"""
    adjustment_formula(P(Y | Do(X)), Z) -> PExpr

The adjustment formula `∑_Z P(Y | X, Z) P(Z)` (`P(Y | X)` when `Z` is
empty). Pass `dag = g` to check first that `Z` is a valid adjustment set.
Other conditioning in the query is not supported.
"""
function adjustment_formula(p::Prob, Z; dag=nothing)
    all(a -> a isa Intervention, p.given) ||
        throw(ArgumentError("adjustment_formula takes P(Y | Do(X)) with interventions only"))
    xs = Atom[a.target for a in p.given]
    isempty(xs) && throw(ArgumentError("adjustment_formula needs an intervention Do(X)"))
    zs = _rvlist(Z)
    if dag !== nothing
        isadjustment(dag, _rvsof(xs), _rvsof(p.targets), zs) ||
            throw(ArgumentError(string("{", join((z.name for z in zs), ", "), "} is not a valid adjustment set")))
    end
    isempty(zs) && return _like(p, p.targets, xs)
    return marg(zs, mul(_like(p, p.targets, vcat(xs, Atom[zs...])), _like(p, Atom[zs...], Atom[])))
end

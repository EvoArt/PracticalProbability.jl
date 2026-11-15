# Identification in linear structural equation models with correlated errors:
# a mixed graph (directed edges i → j with coefficients λᵢⱼ, bidirected edges
# i ↔ j with error covariances) implies Σ = (I - Λ)⁻ᵀ Ω (I - Λ)⁻¹. Which λ are
# determined by Σ, and as what function of it? Instrumental variables
# (conditional IVs, Brito & Pearl 2002) and the half-trek criterion (Foygel,
# Draisma & Drton 2012), which decides generic identifiability of every
# coefficient and gives each as a rational function of the covariances.

"""
    covariance_symbol(a, b) -> RV

The symbolic covariance `σ[a,b]` of two observed variables (the same symbol
for `(b, a)`), as used by [`htc_identify`](@ref) and [`iv_estimand`](@ref).
"""
function covariance_symbol(a::RV, b::RV)
    x, y = isless(b, a) ? (b, a) : (a, b)
    return RV(Symbol("σ[", x.name, ",", y.name, "]"), :continuous)
end
_σ(a::RV, b::RV) = RVal(covariance_symbol(a, b))


"""
    covariance_values(vars, Σ) -> Vector{Pair}

Pairs `σ[a,b] => Σ[i, j]`, to [`evaluate`](@ref) identification formulas at a
covariance matrix.
"""
covariance_values(vars::AbstractVector, Σ::AbstractMatrix) =
    [covariance_symbol(vars[i], vars[j]) => Σ[i, j] for i in eachindex(vars) for j in i:length(vars)]

# Partial covariance σ_{ab·C} = σ_ab - Σ_aC Σ_CC⁻¹ Σ_Cb, symbolically.
function _partialcov(a::RV, b::RV, C::Vector{RV})
    isempty(C) && return _σ(a, b)
    S = RExpr[_σ(x, y) for x in C, y in C]
    adj, det = _radj(S), _rdet(S)
    sa = RExpr[_σ(a, x) for x in C]
    sb = RExpr[_σ(x, b) for x in C]
    q = radd((rmul(sa[i], adj[i, j], sb[j]) for i in eachindex(C), j in eachindex(C))...)
    return _rsub(_σ(a, b), _rdiv(q, det))
end

# The graph with the edge x → y removed.
function _without_edge(g::ADMG, x::RV, y::RV)
    x in parents(g, y) || throw(ArgumentError(string("the graph has no edge ", x.name, " → ", y.name)))
    pa = Dict(v => (v == y ? RV[p for p in ps if p != x] : copy(ps)) for (v, ps) in g.dag.parents)
    return ADMG(DAG(g.dag.nodes, pa), g.bidirected)
end

"""
    isinstrument(g, Z, X, Y; given = RV[]) -> Bool

Whether `Z` is a (conditional) instrumental variable for the edge `X → Y`
given `W = given` (Brito & Pearl 2002): `W` has no descendants of `Y`; `W`
separates `Z` from `Y` once the edge `X → Y` is removed; and `W` does not
separate `Z` from `X`.
"""
function isinstrument(g::Union{DAG,ADMG}, z::RV, x::RV, y::RV; given=RV[])
    G = _asadmg(g)
    W = _rvlist(given)
    (z in W || x in W || y in W || z == x || z == y) && return false
    De = union(descendants(G, y), Set([y]))
    any(w -> w in De, W) && return false
    d_separated(_without_edge(G, x, y), z, y, W) || return false
    return !d_separated(G, z, x, W)
end

"""
    instruments(g, X, Y; given = RV[]) -> Vector{RV}

The variables that are instrumental variables for the edge `X → Y` given
`given` (see [`isinstrument`](@ref)).
"""
instruments(g::Union{DAG,ADMG}, x::RV, y::RV; given=RV[]) =
    RV[z for z in nodes(_asadmg(g)) if isinstrument(g, z, x, y; given)]

"""
    iv_estimand(g, Z, X, Y; given = RV[]) -> RExpr

The coefficient of `X → Y` through the instrument `Z`:
`σ_{ZY·W} / σ_{ZX·W}` in terms of the covariances [`covariance_symbol`](@ref)
(partial covariances given `W`). An `ArgumentError` if `Z` is not an
instrument.
"""
function iv_estimand(g::Union{DAG,ADMG}, z::RV, x::RV, y::RV; given=RV[])
    isinstrument(g, z, x, y; given) ||
        throw(ArgumentError(string(z.name, " is not an instrumental variable for ", x.name, " → ", y.name)))
    W = _rvlist(given)
    return simplify(_rdiv(_partialcov(z, y, W), _partialcov(z, x, W)))
end

# ---------------------------------------------------------------------------
# The half-trek criterion
# ---------------------------------------------------------------------------

"""
    HTCResult

The outcome of [`htc_identify`](@ref): whether every coefficient is
generically identifiable, the coefficients found (`(parent, child) =>`
formula in the covariances), and the variables whose incoming coefficients
the criterion could not determine.
"""
struct HTCResult
    identifiable::Bool
    coefficients::Dict{Tuple{RV,RV},RExpr}
    unsolved::Vector{RV}
    order::Vector{RV}
end
function Base.show(io::IO, ::MIME"text/plain", r::HTCResult)
    println(io, r.identifiable ? "Every coefficient is generically identifiable (half-trek criterion):" :
                "The half-trek criterion leaves the coefficients into " *
                join((v.name for v in r.unsolved), ", ") * " undetermined." *
                (isempty(r.coefficients) ? "" : " Identified:"))
    ks = sort!(collect(keys(r.coefficients)); by=k -> (findfirst(==(k[2]), r.order), findfirst(==(k[1]), r.order)))
    for (i, k) in enumerate(ks)
        print(io, "  λ[", k[1].name, "→", k[2].name, "] = ", _rstr(r.coefficients[k]))
        i == length(ks) || println(io)
    end
end
Base.show(io::IO, r::HTCResult) = print(io, "HTCResult(", r.identifiable ? "identifiable" : "not identified", ")")

# Nodes w reachable from v by a half-trek: v → ⋯ → w, or v ↔ j → ⋯ → w.
function _htr(g::ADMG, v::RV)
    out = descendants(g, v)
    for s in siblings(g, v)
        push!(out, s)
        union!(out, descendants(g, s))
    end
    delete!(out, v)
    return out
end

# A system of half-treks without sided intersection from a subset Y of
# `allowed` onto `targets`, by maximum flow; Y, or `nothing`.
function _halftrek_system(g::ADMG, allowed::Vector{RV}, targets::Vector{RV})
    ns = nodes(g)
    n = length(ns)
    id = Dict(v => i for (i, v) in enumerate(ns))
    # vertices: 1 source, 2 sink; L_in(i), L_out(i), R_in(i), R_out(i)
    Lin(i) = 2 + i; Lout(i) = 2 + n + i; Rin(i) = 2 + 2n + i; Rout(i) = 2 + 3n + i
    N = 2 + 4n
    cap = zeros(Int, N, N)
    for i in 1:n
        cap[Lin(i), Lout(i)] = 1
        cap[Rin(i), Rout(i)] = 1
        cap[Lout(i), Rin(i)] = 1                    # the half-trek starts down from i
    end
    for (a, b) in g.bidirected                       # or crosses one bidirected edge first
        cap[Lout(id[a]), Rin(id[b])] = 1
        cap[Lout(id[b]), Rin(id[a])] = 1
    end
    for v in ns, p in parents(g, v)
        cap[Rout(id[p]), Rin(id[v])] = 1
    end
    for y in allowed
        cap[1, Lin(id[y])] = 1
    end
    for t in targets
        cap[Rout(id[t]), 2] = 1
    end
    flow = _maxflow!(cap, 1, 2)
    flow == length(targets) || return nothing
    # used source edges have no residual capacity left
    return RV[y for y in allowed if cap[1, Lin(id[y])] == 0]
end

# Edmonds–Karp on a residual capacity matrix (modified in place).
function _maxflow!(cap::Matrix{Int}, s::Int, t::Int)
    N = size(cap, 1)
    total = 0
    while true
        prev = zeros(Int, N)
        prev[s] = s
        queue = [s]
        while !isempty(queue) && prev[t] == 0
            u = popfirst!(queue)
            for v in 1:N
                if prev[v] == 0 && cap[u, v] > 0
                    prev[v] = u
                    push!(queue, v)
                end
            end
        end
        prev[t] == 0 && return total
        v = t
        while v != s
            u = prev[v]
            cap[u, v] -= 1
            cap[v, u] += 1
            v = u
        end
        total += 1
    end
end

"""
    htc_identify(g) -> HTCResult

Generic identifiability of the coefficients of a linear SEM on the mixed
graph `g` (an ADMG; a DAG has independent errors) by the recursive half-trek
criterion (Foygel, Draisma & Drton 2012). A variable's incoming coefficients
are solved once there is a set `Y` of variables, none of them the variable
or its siblings, each already solved or not half-trek reachable from it,
with a system of half-treks from `Y` to its parents without sided
intersection. The coefficients come out as rational functions of the
covariances `σ[a,b]` ([`covariance_symbol`](@ref)), by Cramer's rule on
`[(I - Λ)ᵀ Σ]_{Y, pa(v)} λ = [(I - Λ)ᵀ Σ]_{Y, v}`.

Success means every coefficient is identified for almost every parameter
value. Failure is conclusive when the criterion's "HTN" condition holds,
which this does not check: an unsolved variable may still be identifiable
by other means.
"""
function htc_identify(g::Union{DAG,ADMG})
    G = _asadmg(g)
    order = topological_order(G)
    Λ = Dict{Tuple{RV,RV},RExpr}()
    solved = Set{RV}(v for v in order if isempty(parents(G, v)))
    progress = true
    while progress
        progress = false
        for v in order
            v in solved && continue
            pa = sort(parents(G, v); by=p -> findfirst(==(p), order))
            htr = _htr(G, v)
            excl = Set(vcat(v, siblings(G, v)))
            allowed = RV[w for w in order if !(w in excl) && (w in solved || !(w in htr))]
            Y = _halftrek_system(G, allowed, pa)
            Y === nothing && continue
            # rows a_y = [(I - Λ)ᵀ Σ]_{y, ⋅} for solved y in htr(v), Σ_{y, ⋅} otherwise
            row(y, w) = (y in htr && !isempty(parents(G, y))) ?
                _rsub(_σ(y, w), radd((rmul(Λ[(k, y)], _σ(k, w)) for k in parents(G, y))...)) : _σ(y, w)
            A = RExpr[row(y, p) for y in Y, p in pa]
            b = RExpr[row(y, v) for y in Y]
            det = _rdet(A)
            for (j, p) in enumerate(pa)
                Aj = copy(A)
                Aj[:, j] = b
                Λ[(p, v)] = simplify(_rdiv(_rdet(Aj), det))
            end
            push!(solved, v)
            progress = true
        end
    end
    unsolved = RV[v for v in order if !(v in solved)]
    return HTCResult(isempty(unsolved), Λ, unsolved, order)
end

"""
    implied_covariance(g, Λ, Ω) -> Matrix

The covariance `(I - Λ)⁻ᵀ Ω (I - Λ)⁻¹` of a linear SEM on the mixed graph `g`,
with `Λ[i, j]` the coefficient of `i → j` and `Ω` the error covariance, in the
order of `nodes(g)`. Numbers only; for symbolic models without correlated
errors see [`linear_sem`](@ref).
"""
function implied_covariance(g::Union{DAG,ADMG}, Λ::AbstractMatrix, Ω::AbstractMatrix)
    n = length(nodes(_asadmg(g)))
    size(Λ) == (n, n) && size(Ω) == (n, n) || throw(ArgumentError("Λ and Ω must be $(n)×$(n)"))
    B = inv(Matrix{Float64}(_eye(n)) - Λ)
    return B' * Ω * B
end

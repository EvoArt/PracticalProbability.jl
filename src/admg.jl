# Acyclic directed mixed graphs (ADMGs) and the identification of causal
# effects with hidden confounders: the ID and IDC algorithms of Shpitser and
# Pearl (2006), which are complete: when they fail, the hedge they return
# proves that the query is not identifiable.
#
# A bidirected edge X ↔ Y stands for a hidden common cause. For d-separation
# (m-separation) and the do-calculus the ADMG is read as its canonical DAG,
# with one latent node per bidirected edge.

"""
    ADMG(dag::DAG, bidirected)
    ADMG(edges::Pair...; bidirected = [], nodes = RV[])

An acyclic directed mixed graph: a DAG of directed edges plus bidirected
edges `X ↔ Y` (pairs `(X, Y)`) for hidden common causes. Usually built with
[`@admg`](@ref) or by [`latent_projection`](@ref) of a DAG with hidden
variables.
"""
struct ADMG
    dag::DAG
    bidirected::Vector{Tuple{RV,RV}}
    function ADMG(dag::DAG, bi)
        ns = copy(dag.nodes)
        pairs = Tuple{RV,RV}[]
        for (a, b) in bi
            a == b && throw(ArgumentError("a bidirected edge needs two different variables, got $(a.name) ↔ $(a.name)"))
            p = a < b ? (a, b) : (b, a)
            p in pairs || push!(pairs, p)
            a in ns || push!(ns, a)
            b in ns || push!(ns, b)
        end
        d = length(ns) == length(dag.nodes) ? dag : DAG(ns, dag.parents)
        return new(d, pairs)
    end
end
ADMG(edges::Pair...; bidirected=Tuple{RV,RV}[], nodes=RV[]) = ADMG(DAG(edges...; nodes), bidirected)

"""
    @admg begin
        X → M → Y
        X ↔ Y
    end

Build an [`ADMG`](@ref): directed edges as in [`@dag`](@ref), bidirected edges
written `↔` (`\\leftrightarrow<tab>`), with a tuple on either side meaning
every node in it.
"""
macro admg(args...)
    stmts = Any[]
    for a in args
        if a isa Expr && a.head === :block
            append!(stmts, filter(x -> !(x isa LineNumberNode), a.args))
        else
            push!(stmts, a)
        end
    end
    edges = Expr[]
    isolated = Any[]
    bi = Expr[]
    for s in stmts
        if s isa Expr && s.head === :call && s.args[1] === :↔
            ls, rs = _dag_chain(s.args[2]), _dag_chain(s.args[3])
            (length(ls) == 1 && length(rs) == 1) ||
                error("@admg: write bidirected edges as `A ↔ B` or `(A, B) ↔ C`, one per line")
            for l in ls[1], r in rs[1]
                push!(bi, :(($(_rvref(l)), $(_rvref(r)))))
            end
            continue
        end
        chain = _dag_chain(s)
        length(chain) == 1 && append!(isolated, map(_rvref, chain[1]))
        for i in 1:length(chain)-1
            push!(edges, :($(Expr(:tuple, map(_rvref, chain[i])...)) =>
                          $(Expr(:tuple, map(_rvref, chain[i+1])...))))
        end
    end
    return :($ADMG($DAG($(edges...); nodes=$RV[$(isolated...)]), Tuple{$RV,$RV}[$(bi...)]))
end

nodes(g::ADMG) = nodes(g.dag)
parents(g::ADMG, v::RV) = parents(g.dag, v)
children(g::ADMG, v::RV) = children(g.dag, v)
edges(g::ADMG) = edges(g.dag)
ancestors(g::ADMG, vs) = ancestors(g.dag, vs)
descendants(g::ADMG, vs) = descendants(g.dag, vs)
topological_order(g::ADMG) = topological_order(g.dag)

"""
    bidirected(g::ADMG) -> Vector{Tuple{RV,RV}}

The bidirected edges of `g`.
"""
bidirected(g::ADMG) = copy(g.bidirected)

"""
    siblings(g::ADMG, v) -> Vector{RV}

The variables joined to `v` by a bidirected edge.
"""
siblings(g::ADMG, v::RV) = RV[a == v ? b : a for (a, b) in g.bidirected if a == v || b == v]

Base.:(==)(a::ADMG, b::ADMG) = issetequal(a.dag.nodes, b.dag.nodes) && issetequal(edges(a), edges(b)) &&
                               issetequal(a.bidirected, b.bidirected)

function Base.show(io::IO, g::ADMG)
    print(io, "ADMG(", length(g.dag.nodes), " nodes, ", length(edges(g)), " directed, ",
          length(g.bidirected), " bidirected edges)")
end
function Base.show(io::IO, ::MIME"text/plain", g::ADMG)
    println(io, "ADMG with ", length(g.dag.nodes), " nodes, ", length(edges(g)), " directed and ",
            length(g.bidirected), " bidirected edges:")
    order = topological_order(g)
    lines = String[]
    for v in order
        ps = g.dag.parents[v]
        push!(lines, string("  ", v.name, isempty(ps) ? "" : " ← " * join((p.name for p in ps), ", ")))
    end
    for (a, b) in g.bidirected
        push!(lines, string("  ", a.name, " ↔ ", b.name))
    end
    print(io, join(lines, "\n"))
end

"""
    subgraph(g, vs) -> ADMG or DAG

The subgraph induced by the variables `vs`.
"""
function subgraph(g::ADMG, vs)
    s = Set(_rvlist(vs))
    ns = RV[v for v in g.dag.nodes if v in s]
    pa = Dict(v => RV[p for p in g.dag.parents[v] if p in s] for v in ns)
    return ADMG(DAG(ns, pa), Tuple{RV,RV}[(a, b) for (a, b) in g.bidirected if a in s && b in s])
end
function subgraph(g::DAG, vs)
    s = Set(_rvlist(vs))
    ns = RV[v for v in g.nodes if v in s]
    return DAG(ns, Dict(v => RV[p for p in g.parents[v] if p in s] for v in ns))
end

"""
    remove_incoming(g::ADMG, X) -> ADMG

`G[X̄]`: every edge into `X` removed, bidirected edges at `X` included.
"""
function remove_incoming(g::ADMG, vs)
    s = Set(_rvlist(vs))
    return ADMG(remove_incoming(g.dag, vs), Tuple{RV,RV}[(a, b) for (a, b) in g.bidirected if !(a in s) && !(b in s)])
end

"""
    remove_outgoing(g::ADMG, X) -> ADMG

`G[X̲]`: every directed edge out of `X` removed.
"""
remove_outgoing(g::ADMG, vs) = ADMG(remove_outgoing(g.dag, vs), g.bidirected)

"""
    districts(g::ADMG) -> Vector{Vector{RV}}
    c_components(g::ADMG)

The c-components (districts): the sets of variables connected by bidirected
edges. Each variable is in exactly one; variables without bidirected edges are
on their own.
"""
function districts(g::ADMG)
    seen = Set{RV}()
    out = Vector{RV}[]
    for v in topological_order(g)
        v in seen && continue
        comp = RV[]
        stack = RV[v]
        push!(seen, v)
        while !isempty(stack)
            w = pop!(stack)
            push!(comp, w)
            for u in siblings(g, w)
                u in seen || (push!(seen, u); push!(stack, u))
            end
        end
        order = topological_order(g)
        push!(out, sort!(comp; by=x -> findfirst(==(x), order)))
    end
    return out
end
const c_components = districts

_latentname(a::RV, b::RV) = Symbol("U[", a.name, "↔", b.name, "]")

"""
    canonical_dag(g::ADMG) -> DAG

The DAG with one latent node `U[A↔B]` for each bidirected edge `A ↔ B`
(`U[A↔B] → A`, `U[A↔B] → B`). It has the same d-separations among the
observed variables (m-separation in `g`), so the do-calculus on `g` is the
do-calculus on it.
"""
function canonical_dag(g::ADMG)
    isempty(g.bidirected) && return g.dag
    pa = Dict(v => copy(ps) for (v, ps) in g.dag.parents)
    ns = copy(g.dag.nodes)
    for (a, b) in g.bidirected
        u = RV(_latentname(a, b))
        pushfirst!(ns, u)
        pa[u] = RV[]
        push!(pa[a], u)
        push!(pa[b], u)
    end
    return DAG(ns, pa)
end
_asdag(g::ADMG) = canonical_dag(g)
_dags(g::ADMG) = Dict{Symbol,DAG}(:P => canonical_dag(g))

"""
    d_separated(g::ADMG, X, Y, Z = RV[]) -> Bool

m-separation: d-separation in the canonical DAG of `g`.
"""
d_separated(g::ADMG, X, Y, Z=RV[]) = d_separated(canonical_dag(g), X, Y, Z)
holds(s::Independence, g::ADMG) = d_separated(g, s.X, s.Y, s.Z)

simplify(e::PExpr, g::ADMG; kwargs...) = simplify(e; dag=g, kwargs...)

"""
    latent_projection(g::DAG, latent) -> ADMG

The ADMG over the observed variables of `g` (all but `latent`) with the same
causal and independence structure among them (Verma–Pearl): `A → B` when a
directed path from `A` to `B` has only latent variables inside it, and
`A ↔ B` when some latent variable reaches both `A` and `B` by directed paths
through latent variables only.
"""
function latent_projection(g::DAG, latent)
    L = Set(_rvlist(latent))
    for v in L
        v in g.nodes || throw(ArgumentError("$(v.name) is not a node of the graph"))
    end
    obs = RV[v for v in g.nodes if !(v in L)]
    # observed variables reachable from v through latent variables only
    function reach(v)
        out = Set{RV}()
        stack = RV[v]
        seen = Set{RV}([v])
        while !isempty(stack)
            w = pop!(stack)
            for c in children(g, w)
                c in seen && continue
                push!(seen, c)
                c in L ? push!(stack, c) : push!(out, c)
            end
        end
        return out
    end
    pa = Dict(v => RV[] for v in obs)
    for a in obs, b in reach(a)
        push!(pa[b], a)
    end
    bi = Tuple{RV,RV}[]
    for u in L
        r = sort!(collect(reach(u)); by=v -> findfirst(==(v), g.nodes))
        for i in eachindex(r), j in i+1:length(r)
            push!(bi, (r[i], r[j]))
        end
    end
    return ADMG(DAG(obs, pa), bi)
end

# ---------------------------------------------------------------------------
# The ID and IDC algorithms
# ---------------------------------------------------------------------------

"""
    NotIdentifiable

Thrown by `identify` when a causal query is not identifiable from the
observational distribution. `hedge` holds the two variable sets `(F, F′)` of a
hedge (Shpitser–Pearl): two c-forests that agree on the observed distribution
but not on the effect, which proves non-identifiability.
"""
struct NotIdentifiable <: Exception
    query::PExpr
    hedge::Tuple{Vector{RV},Vector{RV}}
end
function Base.showerror(io::IO, e::NotIdentifiable)
    F, F′ = e.hedge
    print(io, "NotIdentifiable: $(e.query) is not identifiable from the observed distribution; ",
          "hedge F = {", _str(F), "}, F′ = {", _str(F′), "}")
end

# The distribution ID works with: `expr === nothing` is the observational
# joint (restricted to the current variables); otherwise an expression for
# the distribution of the current variables. `dos` makes the base an
# experimental distribution P(⋅ | do(dos)) (used by transport and zidentify).
struct _IDDist
    expr::Union{Nothing,PExpr}
    name::Symbol
    dos::Vector{Atom}
end
_IDDist(e, name::Symbol) = _IDDist(e, name, Atom[])

_inorder(order, s) = RV[v for v in order if v in s]

# Σ_{V∖keep} P
function _id_marginal(Pd::_IDDist, keep, V, order)
    Pd.expr === nothing && return isempty(keep) ? Literal(1) : Prob(Atom[_inorder(order, keep)...], Pd.dos, Pd.name)
    return marg(_inorder(order, setdiff(V, keep)), Pd.expr)
end

# P(v | preds) under the current distribution over V
function _id_cond(Pd::_IDDist, v::RV, preds::Vector{RV}, V, order)
    Pd.expr === nothing && return Prob(Atom[v], Atom[preds..., Pd.dos...], Pd.name)
    num = marg(_inorder(order, setdiff(V, union(Set(preds), Set([v])))), Pd.expr)
    den = marg(_inorder(order, setdiff(V, Set(preds))), Pd.expr)
    return simplify(divide(num, den))
end

_preds(order, V, v) = RV[w for w in order[1:findfirst(==(v), order)-1] if w in V]

function _id(y::Set{RV}, x::Set{RV}, Pd::_IDDist, G::ADMG, order::Vector{RV}, q::PExpr)
    V = Set(nodes(G))
    # line 1: no intervention
    isempty(x) && return _id_marginal(Pd, y, V, order)
    # line 2: drop what is not an ancestor of y
    An = union(ancestors(G, collect(y)), y)
    if An != V
        Pn = Pd.expr === nothing ? Pd : _IDDist(marg(_inorder(order, setdiff(V, An)), Pd.expr), Pd.name)
        return _id(y, intersect(x, An), Pn, subgraph(G, collect(An)), order, q)
    end
    # line 3: intervene on what cannot affect y once x is fixed
    gx = remove_incoming(G, collect(x))
    W = setdiff(setdiff(V, x), union(ancestors(gx, collect(y)), y))
    isempty(W) || return _id(y, union(x, W), Pd, G, order, q)
    # line 4: split into the districts of G ∖ x
    Cs = districts(subgraph(G, collect(setdiff(V, x))))
    if length(Cs) > 1
        parts = PExpr[_id(Set(S), setdiff(V, Set(S)), Pd, G, order, q) for S in Cs]
        return marg(_inorder(order, setdiff(V, union(y, x))), mul(parts...))
    end
    S = Set(Cs[1])
    CG = districts(G)
    # line 5: a hedge
    length(CG) == 1 && throw(NotIdentifiable(q, (_inorder(order, V), _inorder(order, S))))
    # line 6: S is a district of G
    if any(C -> Set(C) == S, CG)
        f = mul((_id_cond(Pd, v, _preds(order, V, v), V, order) for v in _inorder(order, S))...)
        return marg(_inorder(order, setdiff(S, y)), f)
    end
    # line 7: S sits inside a larger district S′
    Sp = Set(first(C for C in CG if issubset(S, Set(C))))
    f = mul((_id_cond(Pd, v, _preds(order, V, v), V, order) for v in _inorder(order, Sp))...)
    return _id(y, intersect(x, Sp), _IDDist(f, Pd.name), subgraph(G, collect(Sp)), order, q)
end

function _idc(y::Set{RV}, x::Set{RV}, z::Set{RV}, Pd::_IDDist, G::ADMG, order::Vector{RV}, q::PExpr)
    for zv in _inorder(order, z)
        rest = setdiff(z, Set([zv]))
        gm = remove_outgoing(remove_incoming(G, collect(x)), [zv])
        if d_separated(gm, collect(y), [zv], collect(union(x, rest)))
            return _idc(y, union(x, Set([zv])), rest, Pd, G, order, q)
        end
    end
    p = _id(union(y, z), x, Pd, G, order, q)
    isempty(z) && return p
    return divide(p, marg(_inorder(order, y), p))
end

# Rename summation variables that clash with a variable free around them:
# ∑_x P(x) P(y | m, x) next to P(m | x) becomes ∑_{x′} P(x′) P(y | m, x′).
_prime(v::RV) = RV(Symbol(v.name, "′"), v.kind)
function _freshen(e::PExpr, outer::Set{RV})
    if e isa Marginal
        over = RV[]
        d = Dict{Atom,Atom}()
        taken = union(outer, _allvars(e))
        for v in e.over
            if v in outer
                nv = _prime(v)
                while nv in taken
                    nv = _prime(nv)
                end
                push!(taken, nv)
                d[v] = nv
                push!(over, nv)
            else
                push!(over, v)
            end
        end
        body = isempty(d) ? e.body : _subs(e.body, d)
        return marg(over, _freshen(body, union(outer, Set(over))))
    end
    cs = children(e)
    isempty(cs) && return e
    return rebuild(e, PExpr[_freshen(c, outer) for c in cs])
end

"""
    identify(P(Y | Do(X), Z), g::ADMG; trace = false) -> PExpr
    identify(P(Y | Do(X), Z), g::DAG; latent = [U, …])

An expression for a causal query in terms of the observational distribution,
in a model with hidden confounders (bidirected edges). Runs the IDC algorithm
of Shpitser and Pearl (2006): conditioning variables that can be turned into
interventions (do-calculus rule 2) are, and the rest of the query
`P(Y, Z | do(X)) / P(Z | do(X))` goes through the ID algorithm, which splits
the graph into c-components. The result is simplified, and summation
variables that would clash with free ones are primed (`x′`).

The algorithms are complete: if the query is not identifiable this throws
[`NotIdentifiable`](@ref) with a hedge as the certificate. Use
[`isidentifiable`](@ref) for a yes/no answer.
"""
function identify(p::Prob, g::ADMG; trace::Bool=false)
    _hascf(p) && return _identify_cf(p, g; trace)     # counterfactual.jl
    _check_in_graph(p, g.dag)
    subs = _query_subs(p)
    order = topological_order(g)
    X = Set(_rvsof(a for a in p.given if a isa Intervention))
    Y = Set(_rvsof(p.targets))
    Z = Set(_rvsof(a for a in p.given if !(a isa Intervention)))
    isempty(intersect(Y, union(X, Z))) ||
        throw(ArgumentError("the targets of $(p) also appear after the bar"))
    e = _idc(Y, X, Z, _IDDist(nothing, p.name), g, order, p)
    free = union(Y, X, Z)
    e = substitute(_freshen(e, free), subs...)
    d = derivation(p)
    _push!(d, e, isempty(Z) ? "ID algorithm (Shpitser–Pearl)" : "IDC algorithm (Shpitser–Pearl)")
    apply!(d, x -> simplify(x; trace=true))
    return trace ? d : last(d)
end

"""
    isidentifiable(query, g::ADMG) -> Bool

Whether the causal query `P(Y | Do(X), Z)` is identifiable in `g`.
"""
function isidentifiable(p::Prob, g::ADMG)
    try
        identify(p, g)
        return true
    catch err
        err isa NotIdentifiable && return false
        rethrow()
    end
end
isidentifiable(p::Prob, g::DAG; latent=RV[]) = isidentifiable(p, latent_projection(g, latent))

"""
    hedge(query, g::ADMG) -> Union{Nothing,Tuple{Vector{RV},Vector{RV}}}

The hedge `(F, F′)` that makes the query non-identifiable, or `nothing` if it
is identifiable.
"""
function hedge(p::Prob, g::ADMG)
    try
        identify(p, g)
        return nothing
    catch err
        err isa NotIdentifiable && return err.hedge
        rethrow()
    end
end

# Transportability and selection bias: two ways the data come from somewhere
# other than the population a causal question is about.
#
# * Transportability (Bareinboim & Pearl 2012): experiments were run in a
#   source population; the target population differs in the mechanisms of some
#   variables (a selection diagram marks them) and only passive observations of
#   it are available. `transport` runs the sID algorithm, which is complete.
# * Selection bias (Bareinboim, Tian & Pearl 2014): the data were collected
#   only for units with S = 1, where S depends on some variables. `recover`
#   finds an expression through s-recoverability and generalised adjustment,
#   using unbiased external data on some variables when they are available.

"""
    NotTransportable

Thrown by [`transport`](@ref) when the target effect cannot be computed from
the source experiments and target observations. `witness` holds the variable
sets `(F, F′)` of the s-hedge found.
"""
struct NotTransportable <: Exception
    query::PExpr
    witness::Tuple{Vector{RV},Vector{RV}}
end
Base.showerror(io::IO, e::NotTransportable) =
    print(io, "NotTransportable: $(e.query) cannot be transported: the mechanisms that differ ",
          "reach {", _str(e.witness[2]), "}, which is confounded with the rest of {", _str(e.witness[1]), "}")

struct _TRCtx
    differs::Set{RV}
    source::Symbol
    order::Vector{RV}
    query::PExpr
end

function _sid(y::Set{RV}, x::Set{RV}, Pd::_IDDist, G::ADMG, fixed::Set{RV}, ctx::_TRCtx)
    order = ctx.order
    V = Set(nodes(G))
    isempty(x) && return _id_marginal(Pd, y, V, order)
    An = union(ancestors(G, collect(y)), y)
    if An != V
        Pn = Pd.expr === nothing ? Pd : _IDDist(marg(_inorder(order, setdiff(V, An)), Pd.expr), Pd.name)
        return _sid(y, intersect(x, An), Pn, subgraph(G, collect(An)), fixed, ctx)
    end
    gx = remove_incoming(G, collect(x))
    W = setdiff(setdiff(V, x), union(ancestors(gx, collect(y)), y))
    isempty(W) || return _sid(y, union(x, W), Pd, G, fixed, ctx)
    Cs = districts(subgraph(G, collect(setdiff(V, x))))
    if length(Cs) > 1
        parts = PExpr[_sid(Set(S), setdiff(V, Set(S)), Pd, G, fixed, ctx) for S in Cs]
        return marg(_inorder(order, setdiff(V, union(y, x))), mul(parts...))
    end
    S = Set(Cs[1])
    CG = districts(G)
    if length(CG) == 1
        # not identifiable in the target; the source experiment answers it if
        # no mechanism that differs is in S
        isempty(intersect(S, ctx.differs)) ||
            throw(NotTransportable(ctx.query, (_inorder(order, V), _inorder(order, intersect(S, ctx.differs)))))
        dos = Atom[Intervention(v) for v in _inorder(order, union(x, fixed))]
        return Prob(Atom[_inorder(order, y)...], dos, ctx.source)
    end
    if any(C -> Set(C) == S, CG)
        f = mul((_id_cond(Pd, v, _preds(order, V, v), V, order) for v in _inorder(order, S))...)
        return marg(_inorder(order, setdiff(S, y)), f)
    end
    Sp = Set(first(C for C in CG if issubset(S, Set(C))))
    f = mul((_id_cond(Pd, v, _preds(order, V, v), V, order) for v in _inorder(order, Sp))...)
    return _sid(y, intersect(x, Sp), _IDDist(f, Pd.name), subgraph(G, collect(Sp)),
                union(fixed, setdiff(V, Sp)), ctx)
end

"""
    transport(P(Y | Do(X)), g::ADMG, differs; source = :π, trace = false) -> PExpr
    transport(P(Y | Do(X)), g::DAG, differs; latent = [], …)

Transport a causal effect from a source population, where experiments can be
run, to a target population, where only observations are available
(Bareinboim & Pearl's sID algorithm). `differs` lists the variables whose
mechanism (or exogenous causes) may differ between the two: the selection
diagram has an S-node pointing into each. The result uses target observational
terms `P(⋯)` and source experimental terms `π(⋯ | do(⋯))` (named by `source`).
The algorithm is complete: throws [`NotTransportable`](@ref) when the effect
cannot be transported. Effects identifiable in the target alone come back in
target terms only.
"""
function transport(p::Prob, g::ADMG, differs; source::Symbol=:π, trace::Bool=false)
    _check_in_graph(p, g.dag)
    any(a -> !(a isa Intervention), p.given) &&
        throw(ArgumentError("transport takes P(Y | Do(X)); condition afterwards"))
    D = Set(_rvlist(differs))
    for v in D
        v in g.dag.nodes || throw(ArgumentError("$(v.name) is not a node of the graph"))
    end
    order = topological_order(g)
    X = Set(_rvsof(p.given))
    Y = Set(_rvsof(p.targets))
    e = _sid(Y, X, _IDDist(nothing, p.name), g, Set{RV}(), _TRCtx(D, source, order, p))
    e = substitute(_freshen(e, union(X, Y)), _query_subs(p)...)
    d = derivation(p)
    _push!(d, e, "sID algorithm (Bareinboim–Pearl)")
    apply!(d, x -> simplify(x; trace=true))
    return trace ? d : last(d)
end
transport(p::Prob, g::DAG, differs; latent=RV[], kwargs...) =
    transport(p, latent_projection(g, latent), differs; kwargs...)

"""
    istransportable(query, g, differs) -> Bool
"""
function istransportable(p::Prob, g, differs; kwargs...)
    try
        transport(p, g, differs; kwargs...)
        return true
    catch err
        err isa NotTransportable && return false
        rethrow()
    end
end

# ---------------------------------------------------------------------------
# Selection bias
# ---------------------------------------------------------------------------

"""
    NotRecoverable

Thrown by [`recover`](@ref) when no expression of the supported forms recovers
the query from the selection-biased (and external) data.
"""
struct NotRecoverable <: Exception
    query::PExpr
    reason::String
end
Base.showerror(io::IO, e::NotRecoverable) = print(io, "NotRecoverable: $(e.query): ", e.reason)

function _candidate_sets(cands::Vector{RV}, maxsize::Integer)
    out = Vector{RV}[]
    for k in 0:min(maxsize, length(cands))
        append!(out, _combinations(cands, k))
    end
    return out
end

"""
    recover(P(Y | X), g::DAG, S; external = [], measured = nothing, latent = [], maxsize = 3) -> PExpr
    recover(P(Y | Do(X)), g::DAG, S; …)

Recover a query about the whole population from data selected on the event
`S` (a node of `g` with no children, declared with `@event S`; its parents are
what selection depends on). Terms conditioned on `S` come from the biased
data, which covers the variables in `measured` (default: all observed ones);
terms without `S` come from `external` unbiased data on the variables listed.
`latent` lists hidden variables of `g`.

Conditional distributions `P(Y | X)` (Bareinboim, Tian & Pearl 2014):

- without external data, `P(Y | X)` is recoverable **if and only if**
  `Y ⟂ S | X`, as `P(Y | X, S)` (Theorem 1), so a failure here is final;
- with external data on `X` and some `C`: `∑_c P(Y | X, c, S) P(c | X)` when
  `(C, X)` separates `S` from `Y`, decided by one d-separation test over the
  largest admissible `C` (Theorems 2–3), then shrunk to a minimal set;
- with external data on `Y` but not all of `X`: through `P(X | Y)` and Bayes'
  rule (Corollaries 3–4); or through the joint `P(X, Y)` (Corollary 5).

Causal effects `P(Y | Do(X))`:

- in a model without hidden variables, recoverable without external data **if
  and only if** no variable in `D = An(Y)` of `G[V ∖ X]` is an ancestor of `S`,
  as `∑_{D∖Y} ∏_{v∈D} P(v | pa(v), S)` (Bareinboim & Tian 2015, Theorem 2);
- the selection-backdoor criterion (BTP 2014, Theorem 5): `∑_z P(Y | X, z, S) P(z)`;
- with hidden variables, the RC algorithm (Bareinboim & Tian 2015, Fig. 4),
  which recovers the c-components that contain no ancestor of `S`, peels them
  off and identifies the query from them;
- finally the query is identified by ID and each term recovered on its own,
  using external data where needed (front-door-like cases).

Throws [`NotRecoverable`](@ref) when nothing applies; the reason says when the
criterion that failed is complete (no method can recover the query).
"""
function recover(p::Prob, g::DAG, S::RV; external=RV[], measured=nothing, maxsize::Integer=3, latent=RV[])
    isevent(S) || throw(ArgumentError("the selection variable must be an event (@event S)"))
    S in g.nodes || throw(ArgumentError("$(S.name) is not a node of the graph"))
    isempty(children(g, S)) || throw(ArgumentError("the selection node $(S.name) must have no children"))
    _check_in_graph(p, g)
    subs = _query_subs(p)
    ext = Set(_rvlist(external))
    L = Set(_rvlist(latent))
    M = measured === nothing ? Set(RV[v for v in g.nodes if v != S && !(v in L)]) : Set(_rvlist(measured))
    Y = _rvsof(p.targets)
    X = _rvsof(a for a in p.given if a isa Intervention)
    C = _rvsof(a for a in p.given if !(a isa Intervention))
    (isempty(X) || isempty(C)) || throw(ArgumentError("recover takes P(Y | X) or P(Y | Do(X))"))
    name = p.name
    pr(t, gv) = Prob(Atom[t...], Atom[gv...], name)
    finish(e) = simplify(substitute(_freshen(e, Set(vcat(Y, X, C))), subs...))
    okdata(e) = e !== nothing && _data_ok(e, g, S, M, ext)
    cands = RV[v for v in topological_order(g) if !(v in Y) && !(v in X) && !(v in C) && v != S && !(v in L)]

    if isempty(X)                       # P(Y | C)
        YC = union(Set(Y), Set(C))
        if d_separated(g, Y, S, C) && issubset(YC, M)
            return finish(pr(Y, vcat(C, S)))                                   # Theorem 1
        end
        TM = intersect(ext, M)
        an = union(ancestors(g, vcat(Y, C, [S])), Set(vcat(Y, C)))
        Cstar = RV[v for v in topological_order(g) if v in TM && v in an && !(v in YC) && v != S]
        # Theorems 2–3: external data on C (the conditioning) and a separator
        if issubset(Set(C), ext) && issubset(YC, M) && d_separated(g, Y, S, vcat(C, Cstar))
            W = _minimise(Cstar, w -> d_separated(g, Y, S, vcat(C, w)))
            e = isempty(W) ? pr(Y, vcat(C, S)) : marg(W, mul(pr(Y, vcat(C, W, S)), pr(W, C)))
            return finish(e)
        end
        # Corollaries 3–4: external data on Y, not on all of the conditioning
        X0 = RV[v for v in C if v in ext]
        Xm = RV[v for v in C if !(v in ext)]
        if issubset(Set(Y), ext) && !isempty(Xm) && issubset(YC, M) &&
           d_separated(g, Xm, S, vcat(Cstar, Y, X0))
            W = _minimise(Cstar, w -> d_separated(g, Xm, S, vcat(w, Y, X0)))
            pxm = isempty(W) ? pr(Xm, vcat(Y, X0, S)) : marg(W, mul(pr(Xm, vcat(Y, X0, W, S)), pr(W, vcat(Y, X0))))
            num = mul(pxm, pr(Y, X0))
            return finish(divide(num, marg(Y, num)))
        end
        # Corollary 5: through the joint P(Y, C)
        if !isempty(C) && issubset(YC, M) && d_separated(g, vcat(Y, C), S, Cstar)
            W = _minimise(Cstar, w -> d_separated(g, vcat(Y, C), S, w))
            pyx = isempty(W) ? pr(vcat(Y, C), [S]) : marg(W, mul(pr(vcat(Y, C), vcat(W, S)), pr(W, RV[])))
            return finish(divide(pyx, marg(Y, pyx)))
        end
        r = _recover_by_id(p, g, S, ext, L, maxsize)
        okdata(r) && return r
        throw(NotRecoverable(p, isempty(ext) ?
            "$(_str(Y)) and $(S.name) are not d-separated given $(_str(C)); without external data that is " *
            "necessary (Bareinboim–Tian–Pearl 2014, Theorem 1), so no method recovers it" :
            "no separator measured in the population works (BTP 2014, Theorems 2–3, Corollaries 3–5), " *
            "and identifying it term by term does not either"))
    end

    # P(Y | Do(X)). Without hidden variables, Bareinboim–Tian Theorem 2 decides it.
    markovian = isempty(L)
    if markovian
        e, culprit = _recover_markovian(Y, X, g, S, name)
        if e !== nothing && okdata(e)
            return finish(e)
        end
        if isempty(ext) && culprit !== nothing
            throw(NotRecoverable(p, "$(culprit.name) is an ancestor of both $(_str(Y)) (with $(_str(X)) removed) " *
                                    "and $(S.name); without external data no method recovers the effect " *
                                    "(Bareinboim–Tian 2015, Theorem 2, which is complete here)"))
        end
    end
    # the selection-backdoor criterion (BTP 2014, Theorem 5)
    desc = descendants(g, X)
    gx = remove_outgoing(g, X)
    for Z in _candidate_sets(cands, maxsize)
        Zp = RV[v for v in Z if !(v in desc)]
        Zm = RV[v for v in Z if v in desc]
        d_separated(gx, Y, X, Zp) || continue                                     # (i)
        (isempty(Zm) || d_separated(g, Zm, Y, vcat(X, Zp))) || continue          # (ii)
        d_separated(g, Y, S, vcat(X, Z)) || continue                              # (iii)
        issubset(union(Set(Z), Set(X), Set(Y)), M) || continue                    # (iv)
        isempty(Z) && return finish(pr(Y, vcat(X, S)))
        pz = issubset(Set(Z), ext) ? pr(Z, RV[]) : d_separated(g, Z, S) ? pr(Z, [S]) : nothing
        pz === nothing && continue
        return finish(marg(Z, mul(pr(Y, vcat(X, Z, S)), pz)))
    end
    # the RC algorithm (hidden variables, biased data only)
    r = _recover_rc(p, Y, X, g, S, L)
    if r !== nothing && okdata(r)
        return finish(r)
    end
    r = _recover_by_id(p, g, S, ext, L, maxsize)
    okdata(r) && return r
    throw(NotRecoverable(p, "neither the selection-backdoor criterion, the RC algorithm (Bareinboim–Tian 2015) " *
                            "nor term-by-term recovery of an identified expression applies with the data available"))
end

# Drop elements of a separating set while it still separates.
function _minimise(W::Vector{RV}, test)
    W = copy(W)
    for v in reverse(copy(W))
        rest = RV[w for w in W if w != v]
        test(rest) && (W = rest)
    end
    return W
end

# Does every term come from the data at hand? Terms with S from the biased data
# (over `M`), the others from the external data (over `T`).
function _data_ok(e::PExpr, g::DAG, S::RV, M::Set{RV}, T::Set{RV})
    ok = Ref(true)
    mapprobs(e) do t
        vs = RV[]
        for v in _rvsof(vcat(t.targets, t.given))
            v == S && continue
            if !(v in g.nodes)
                b = _unprimed(v)
                v = b === nothing ? v : b
            end
            push!(vs, v)
        end
        biased = any(a -> a == S, t.given)
        any(a -> a isa Intervention, t.given) && (ok[] = false)
        ok[] &= issubset(Set(vs), biased ? M : T)
        return t
    end
    return ok[]
end

# Bareinboim–Tian 2015, Theorem 2 (no hidden variables): with D = An(Y) in
# G[V ∖ X], the effect is ∑_{D∖Y} ∏_{v∈D} P(v | pa(v), S) when no v ∈ D is an
# ancestor of S, and not recoverable otherwise. Returns (expression, nothing)
# or (nothing, an offending variable).
function _recover_markovian(Y, X, g::DAG, S::RV, name::Symbol)
    gv = subgraph(g, RV[v for v in g.nodes if !(v in X) && v != S])
    D = union(ancestors(gv, Y), Set(Y))
    anS = ancestors(g, S)
    order = topological_order(g)
    for v in order
        v in D && v in anS && return (nothing, v)
    end
    f = mul((Prob(Atom[v], Atom[g.parents[v]..., S], name) for v in order if v in D)...)
    return (marg(RV[v for v in order if v in D && !(v in Y)], f), nothing)
end

# The RC algorithm of Bareinboim & Tian (2015, Fig. 4) for P_x(y): with
# D = An(Y) in G[V ∖ X] and its c-components Dᵢ, P_x(y) = ∑_{D∖Y} ∏ Q[Dᵢ], each
# Q[Dᵢ] recovered by RC. Returns nothing when RC fails.
function _recover_rc(p::Prob, Y, X, g::DAG, S::RV, L::Set{RV})
    A = latent_projection(g, collect(L))
    isempty(siblings(A, S)) || return nothing          # S must not share hidden causes
    V = RV[v for v in topological_order(A) if v != S]
    GV = subgraph(A, RV[v for v in V if !(v in X)])
    D = union(ancestors(GV, Y), Set(Y))
    parts = PExpr[]
    try
        for Di in districts(subgraph(A, collect(D)))
            push!(parts, _rc(Set(Di), _IDDist(nothing, p.name, Atom[S]), A, S, p))
        end
    catch err
        err isa NotRecoverable && return nothing
        rethrow()
    end
    return marg(RV[v for v in V if v in D && !(v in Y)], mul(parts...))
end

# RC(D, P, G): Q[D] for a c-component set D, from the (unnormalised) function
# P of the biased data. Ancestors of S come first in the order used for the
# c-component factors, so P(S = 1 | pa_S) cancels from their ratios (Lemma 3).
function _rc(D::Set{RV}, Pd::_IDDist, G::ADMG, S::RV, q::PExpr)
    V = Set(RV[v for v in nodes(G) if v != S])
    # 1. drop what is neither an ancestor of D nor of S
    keep = union(ancestors(G, collect(D)), D, ancestors(G, S))
    if !issubset(V, keep)
        order0 = topological_order(G)
        drop = _inorder(order0, setdiff(V, keep))
        Pn = Pd.expr === nothing ? Pd : _IDDist(marg(drop, Pd.expr), Pd.name, Pd.dos)
        return _rc(D, Pn, subgraph(G, vcat(collect(intersect(V, keep)), [S])), S, q)
    end
    anS = ancestors(G, S)
    topo = RV[v for v in topological_order(G) if v != S]
    order = vcat(RV[v for v in topo if v in anS], RV[v for v in topo if !(v in anS)])
    # 2.–3. the c-components free of ancestors of S are recoverable (Lemma 3)
    comps = districts(subgraph(G, collect(V)))
    Cs = [Set(c) for c in comps if isempty(intersect(Set(c), anS))]
    isempty(Cs) && throw(NotRecoverable(q, "every c-component contains an ancestor of $(S.name)"))
    Q(C) = mul((_id_cond(Pd, v, _preds(order, V, v), V, order) for v in _inorder(order, C))...)
    # 4. D inside one of them: identify Q[D] from Q[Cᵢ]
    for Ci in Cs
        issubset(D, Ci) || continue
        try
            return _id(D, setdiff(Ci, D), _IDDist(Q(Ci), Pd.name), subgraph(G, collect(Ci)), order, q)
        catch err
            err isa NotIdentifiable && throw(NotRecoverable(q, "Q[$(_str(collect(D)))] is not identifiable from Q[$(_str(collect(Ci)))]"))
            rethrow()
        end
    end
    # 5. divide them out and recurse on the rest
    Call = reduce(union, Cs)
    joint = Pd.expr === nothing ? Prob(Atom[_inorder(order, V)...], Pd.dos, Pd.name) : Pd.expr
    Pn = _IDDist(divide(joint, mul((Q(Ci) for Ci in Cs)...)), Pd.name)
    rest = setdiff(V, Call)
    return _rc(D, Pn, subgraph(G, vcat(collect(rest), [S])), S, q)
end

# The general route: identify the query (hidden variables allowed) and recover
# each observational term on its own: P(T | C) is P(T | C, S) when T ⟂ S | C,
# comes from external data when they cover T ∪ C, or is ∑_W P(T | C, W, S) P(W | C)
# for a W that separates T from S with external data on C ∪ W.
function _recover_by_id(p::Prob, g::DAG, S::RV, ext::Set{RV}, L::Set{RV}, maxsize)
    isempty(children(g, S)) || return nothing
    gs = subgraph(g, RV[v for v in g.nodes if v != S])
    e = try
        identify(p, latent_projection(gs, collect(L)))::PExpr
    catch err
        err isa NotIdentifiable && return nothing
        rethrow()
    end
    ok = Ref(true)
    out = mapprobs(e) do t
        r = _recover_term(t, g, S, ext, L, maxsize)
        r === nothing && (ok[] = false; return t)
        return r
    end
    ok[] || return nothing
    free = Set{RV}(_rvsof(vcat(p.targets, p.given)))
    return simplify(_freshen(out, free))::PExpr
end

function _recover_term(t::Prob, g::DAG, S::RV, ext::Set{RV}, L::Set{RV}, maxsize)
    any(a -> a isa Intervention, t.given) && return nothing
    base = Dict{Atom,Atom}()
    for v in _rvsof(vcat(t.targets, t.given))
        v in g.nodes && continue
        b = _unprimed(v)
        b !== nothing && b in g.nodes && (base[v] = b)
    end
    t0 = isempty(base) ? t : _subs(t, base)
    t0 isa Prob || return nothing
    T, C = _rvsof(t0.targets), _rvsof(t0.given)
    d_separated(g, T, S, C) && return Prob(t.targets, Atom[t.given..., S], t.name)
    issubset(union(Set(T), Set(C)), ext) && return t
    cands = RV[v for v in topological_order(g) if !(v in T) && !(v in C) && v != S && !(v in L)]
    for W in _candidate_sets(cands, maxsize)
        isempty(W) && continue
        d_separated(g, T, S, vcat(C, W)) || continue
        issubset(union(Set(C), Set(W)), ext) || continue
        return marg(W, mul(Prob(t.targets, Atom[t.given..., W..., S], t.name), Prob(Atom[W...], t.given, t.name)))
    end
    return nothing
end

"""
    isrecoverable(query, g, S; kwargs...) -> Bool
"""
function isrecoverable(p::Prob, g::DAG, S::RV; kwargs...)
    try
        recover(p, g, S; kwargs...)
        return true
    catch err
        err isa NotRecoverable && return false
        rethrow()
    end
end

# ---------------------------------------------------------------------------
# Several domains with limited experiments: TR^mz and z-identifiability
# ---------------------------------------------------------------------------

"""
    Domain(name; differs = RV[], experiments = [])

A source population for [`transport`](@ref): `name` names its distribution in
the result (`π₁(⋯ | do(⋯))`), `differs` lists the variables whose mechanism
differs from the target's (S-nodes point into them), and `experiments` lists
the sets of variables that were intervened on there (`[[X], [X, Z]]`), or
`:all` when any experiment can be run.
"""
struct Domain
    name::Symbol
    differs::Set{RV}
    experiments::Union{Symbol,Vector{Vector{RV}}}
end
function Domain(name::Symbol; differs=RV[], experiments=Vector{RV}[])
    exps = experiments === :all ? :all : Vector{RV}[_rvlist(z) for z in experiments]
    return Domain(name, Set(_rvlist(differs)), exps)
end

struct _MZCtx
    domains::Vector{Domain}           # the target first (name of the query, nothing differs)
    order::Vector{RV}
    query::PExpr
    G::ADMG                           # the whole graph
end


# Is the effect of x on y in the current graph the same in domain d as in the
# target? (S-nodes into the differing variables are d-separated from y given x
# once x is intervened on.)
function _same_effect(d::Domain, y::Set{RV}, x::Set{RV}, G::ADMG)
    # intervening on x cuts the S-nodes into x too
    D = setdiff(intersect(d.differs, Set(nodes(G))), x)
    isempty(D) && return true
    g = canonical_dag(remove_incoming(G, collect(x)))
    ns = copy(g.nodes)
    pa = Dict(v => copy(ps) for (v, ps) in g.parents)
    S = RV[]
    for v in D
        s = RV(Symbol("S[", v.name, "]"))
        push!(ns, s); pa[s] = RV[]; push!(pa[v], s); push!(S, s)
    end
    return d_separated(DAG(ns, pa), S, collect(y), collect(x))
end

function _trmz(y::Set{RV}, x::Set{RV}, Pd::_IDDist, G::ADMG, fixed::Set{RV}, ctx::_MZCtx)
    order = ctx.order
    V = Set(nodes(G))
    isempty(x) && return _id_marginal(Pd, y, V, order)
    An = union(ancestors(G, collect(y)), y)
    if An != V
        Pn = Pd.expr === nothing ? Pd : _IDDist(marg(_inorder(order, setdiff(V, An)), Pd.expr), Pd.name)
        return _trmz(y, intersect(x, An), Pn, subgraph(G, collect(An)), fixed, ctx)
    end
    gx = remove_incoming(G, collect(x))
    W = setdiff(setdiff(V, x), union(ancestors(gx, collect(y)), y))
    isempty(W) || return _trmz(y, union(x, W), Pd, G, fixed, ctx)
    Cs = districts(subgraph(G, collect(setdiff(V, x))))
    if length(Cs) > 1
        parts = PExpr[_trmz(Set(S), setdiff(V, Set(S)), Pd, G, fixed, ctx) for S in Cs]
        return marg(_inorder(order, setdiff(V, union(y, x))), mul(parts...))
    end
    S = Set(Cs[1])
    CG = districts(G)
    if length(CG) == 1
        # A hedge for the distribution at hand. What is needed is P_{x, κ}(y)
        # (κ: variables fixed on the way here). An experiment on Z ⊆ x ∪ κ
        # (touching x) in the target, or in a domain where that effect is the
        # same, starts a fresh identification of it from P_Z on the graph
        # without Z.
        xk = union(x, fixed)
        G0 = ctx.G
        for d in ctx.domains
            _same_effect(d, y, xk, G0) || continue
            cands = d.experiments === :all ? [xk] : [Set(z) for z in d.experiments]
            for Z in cands
                (issubset(Z, xk) && !isempty(intersect(Z, x))) || continue
                dos = Atom[Intervention(v) for v in _inorder(order, Z)]
                rest = setdiff(Set(nodes(G0)), Z)
                try
                    return _trmz(y, setdiff(xk, Z), _IDDist(nothing, d.name, dos), subgraph(G0, collect(rest)),
                                 Set{RV}(), ctx)
                catch err
                    err isa NotTransportable || rethrow()
                end
            end
        end
        throw(NotTransportable(ctx.query, (_inorder(order, V), _inorder(order, S))))
    end
    if any(C -> Set(C) == S, CG)
        f = mul((_id_cond(Pd, v, _preds(order, V, v), V, order) for v in _inorder(order, S))...)
        return marg(_inorder(order, setdiff(S, y)), f)
    end
    Sp = Set(first(C for C in CG if issubset(S, Set(C))))
    f = mul((_id_cond(Pd, v, _preds(order, V, v), V, order) for v in _inorder(order, Sp))...)
    return _trmz(y, intersect(x, Sp), _IDDist(f, Pd.name, Pd.dos), subgraph(G, collect(Sp)),
                 union(fixed, setdiff(V, Sp)), ctx)
end

"""
    transport(P(Y | Do(X)), g::ADMG; sources = Domain[], experiments = [], trace = false)
    transport(P(Y | Do(X)), g::DAG; latent = [], …)

Transport from several source domains with limited experiments
(Bareinboim & Pearl's TR^mz): the target offers its observational
distribution `P` and experiments on the sets in `experiments`; each
[`Domain`](@ref) in `sources` offers experiments on its sets, valid in the
target where its differing mechanisms do not reach the effect. The result uses
`P(⋯)`, `P(⋯ | do(⋯))` for target experiments and `π(⋯ | do(⋯))` for a source
named `π`. Throws [`NotTransportable`](@ref) when no combination works.

With no sources this decides z-identifiability (see [`zidentify`](@ref)).
"""
function transport(p::Prob, g::ADMG; sources=Domain[], experiments=Vector{RV}[], trace::Bool=false)
    _check_in_graph(p, g.dag)
    any(a -> !(a isa Intervention), p.given) &&
        throw(ArgumentError("transport takes P(Y | Do(X)); condition afterwards"))
    target = Domain(p.name; experiments)
    for d in sources, v in d.differs
        v in g.dag.nodes || throw(ArgumentError("$(v.name) is not a node of the graph"))
    end
    order = topological_order(g)
    X = Set(_rvsof(p.given))
    Y = Set(_rvsof(p.targets))
    ctx = _MZCtx(Domain[target, sources...], order, p, g)
    e = _trmz(Y, X, _IDDist(nothing, p.name), g, Set{RV}(), ctx)
    e = substitute(_freshen(e, union(X, Y)), _query_subs(p)...)
    d = derivation(p)
    _push!(d, e, "TRᵐᶻ algorithm (Bareinboim–Pearl)")
    apply!(d, x -> simplify(x; trace=true))
    return trace ? d : last(d)
end
transport(p::Prob, g::DAG; latent=RV[], kwargs...) = transport(p, latent_projection(g, latent); kwargs...)

"""
    zidentify(P(Y | Do(X)), g, Z; trace = false) -> PExpr

z-identifiability (Bareinboim & Pearl 2012): identify the effect from the
observational distribution together with experiments on any subset of the
surrogate variables `Z` (the result has terms `P(⋯ | do(z))`). Throws
[`NotTransportable`](@ref) when that is not enough.
"""
function zidentify(p::Prob, g::ADMG, Z; kwargs...)
    zs = _rvlist(Z)
    subsets = Vector{RV}[z for k in 1:length(zs) for z in _combinations(zs, k)]
    return transport(p, g; experiments=subsets, kwargs...)
end
zidentify(p::Prob, g::DAG, Z; latent=RV[], kwargs...) = zidentify(p, latent_projection(g, latent), Z; kwargs...)

"""
    iszidentifiable(query, g, Z) -> Bool
"""
function iszidentifiable(p::Prob, g, Z; kwargs...)
    try
        zidentify(p, g, Z; kwargs...)
        return true
    catch err
        err isa NotTransportable && return false
        rethrow()
    end
end

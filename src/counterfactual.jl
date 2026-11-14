# Counterfactuals: potential outcomes `Y_x` as atoms, their exact computation
# in a structural causal model through the parallel-worlds network, and their
# identification from the observational distribution by the ID* and IDC*
# algorithms of Shpitser and Pearl (2007), built on make-cg (the
# counterfactual graph).

# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------

"""
    Counterfactual(rv, sub, value)
    Y(Do(X(0)))        # Y_{X=0}, the potential outcome
    Y(Do(X(0)))(1)     # the event Y_{X=0} = 1
    Y(Do(X(0), Z(1)))  # Y_{X=0, Z=1}

The value `Y` would have taken had `X` been set to `0` (a potential outcome),
and events on it. They go inside `P(⋯)` with ordinary events, as in the
effect of treatment on the treated `P(Y(Do(X(1)))(1) | X(0))`. Interventions
need values. [`compute`](@ref) evaluates counterfactuals in a structural causal
model (see [`rand_scm`](@ref)); [`identify`](@ref) expresses them in
observational terms when the graph allows it.
"""
struct Counterfactual <: Atom
    rv::RV
    sub::Vector{Atom}          # Event X(x), or a nested potential outcome M_{x′} (no value)
    value::Any
    hasvalue::Bool
end

_subrv(e::Event) = e.rv
_subrv(c::Counterfactual) = c.rv

function _cfsub(is)
    evs = Atom[]
    for i in is
        e = if i isa Intervention
            i.target isa Event || throw(ArgumentError(
                "give the intervention in a potential outcome a value: Y(Do(X(0))), not Y(Do(X))"))
            i.target
        elseif i isa Counterfactual && !i.hasvalue
            i                                       # nested: set M to its value M_{x′}
        else
            throw(ArgumentError("a potential outcome is written Y(Do(X(value))) or, nested, Y(Do(X(1)), M(Do(X(0))))"))
        end
        any(x -> _subrv(x) == _subrv(e), evs) && throw(ArgumentError("$(_subrv(e).name) is intervened on twice"))
        push!(evs, e)
    end
    return sort!(evs; by=_subrv)
end
(x::RV)(i::Union{Intervention,Counterfactual}, more::Union{Intervention,Counterfactual}...) =
    Counterfactual(x, _cfsub((i, more...)), nothing, false)
(x::RV)(is::Tuple{Vararg{Intervention}}) = Counterfactual(x, _cfsub(is), nothing, false)
function (c::Counterfactual)(v)
    c.hasvalue && throw(ArgumentError("$(c) already has a value"))
    return Counterfactual(c.rv, c.sub, v, true)
end

Base.:(==)(a::Counterfactual, b::Counterfactual) =
    a.rv == b.rv && a.sub == b.sub && a.hasvalue == b.hasvalue && isequal(a.value, b.value)
Base.hash(a::Counterfactual, h::UInt) = hash(a.value, hash(a.sub, hash(a.rv, hash(:Counterfactual, h))))

rv(c::Counterfactual) = c.rv
rvs(c::Counterfactual) = RV[c.rv]
freerv(::Counterfactual) = nothing
iseventlike(c::Counterfactual) = c.hasvalue
_isnested(c::Counterfactual) = any(e -> e isa Counterfactual, c.sub)

_subitem(e::Event) = string(e.rv.name, "=", _valstr(e.value))
_subitem(c::Counterfactual) = _astr(c)
_subtext(c::Counterfactual) = join(map(_subitem, c.sub), ",")
function _astr(c::Counterfactual, inner=false)
    s = string(c.rv.name, "_{", _subtext(c), "}")
    c.hasvalue || return s
    s = string(s, "=", _valstr(c.value))
    return inner ? string("{", s, "}") : s
end
_lxsubitem(e::Event) = string(_lx(e.rv.name), " = ", _lx(e.value))
_lxsubitem(c::Counterfactual) = _lxatom(c)
function _lxatom(c::Counterfactual, inner=false)
    s = string(_lx(c.rv.name), "_{", join(map(_lxsubitem, c.sub), ", "), "}")
    c.hasvalue && (s = string(s, " = ", _lx(c.value)))
    return inner && c.hasvalue ? string("\\{", s, "\\}") : s
end

_hascf(p::Prob) = any(a -> a isa Counterfactual, vcat(p.targets, p.given))

# ---------------------------------------------------------------------------
# Exact computation: the parallel-worlds network of a structural model
# ---------------------------------------------------------------------------

"""
    rand_scm(g::DAG; card = 2, noise = 3, rng) -> BayesNet

A random structural causal model on `g`: every variable with parents is a
deterministic function of its parents and its own exogenous noise variable
`U_v` (with `noise` values), and every root has a random distribution. The
result is an ordinary [`BayesNet`](@ref) (with the noise variables as extra
roots), so observational and interventional queries compute as usual, and
counterfactual queries compute through the parallel-worlds network.
"""
function rand_scm(g::DAG; card::Integer=2, noise::Integer=3, rng::AbstractRNG=Random.default_rng())
    specs = Pair[]
    n(v) = isevent(v) ? 2 : card
    for v in topological_order(g)
        ps = g.parents[v]
        if isempty(ps)
            T = rand(rng, n(v)) .+ 0.05
            push!(specs, v => T ./ sum(T))
        else
            u = RV(Symbol("U_", v.name))
            u in g.nodes && throw(ArgumentError("the graph already has a variable named $(u.name)"))
            w = rand(rng, noise) .+ 0.05
            push!(specs, u => w ./ sum(w))
            noise >= n(v) || throw(ArgumentError("noise must have at least as many values as $(v.name)"))
            dims = (n(v), map(n, ps)..., noise)
            T = zeros(dims)
            # every value is reachable from every parent configuration (positivity)
            for idx in CartesianIndices(dims[2:end-1])
                outs = Random.shuffle(rng, vcat(Random.randperm(rng, n(v)), rand(rng, 1:n(v), noise - n(v))))
                for k in 1:noise
                    T[outs[k], Tuple(idx)..., k] = 1.0
                end
            end
            push!(specs, Given(Atom[v], Atom[ps..., u]) => T)
        end
    end
    return BayesNet(specs...)
end

_worldname(v::RV, sub::Vector{Atom}) =
    isempty(sub) ? v : RV(Symbol(v.name, "_{", join(map(_subitem, sub), ","), "}"), v.kind)

# Turn do(⋅) in the conditioning into subscripts: P(y | do(x), z) = P(y_x | z_x).
# Plain variables become potential outcomes without a value (free).
function _cf_atoms(p::Prob)
    dos = Event[]
    for a in p.given
        a isa Intervention || continue
        a.target isa Event || throw(ArgumentError("give do(⋅) a value in a counterfactual query"))
        push!(dos, a.target)
    end
    function lift(a)
        a isa Intervention && return nothing
        isempty(dos) && return a
        base, val, has, sub = if a isa Counterfactual
            (a.rv, a.value, a.hasvalue, a.sub)
        elseif a isa Event
            (a.rv, a.value, true, Atom[])
        elseif a isa RV && !isevent(a)
            (a, nothing, false, Atom[])
        else
            throw(ArgumentError("counterfactual queries take variables and events X(x); got $(_astr(a))"))
        end
        newsub = copy(sub)
        for d in dos
            any(e -> _subrv(e) == d.rv, newsub) || push!(newsub, d)
        end
        return Counterfactual(base, sort!(newsub; by=_subrv), val, has)
    end
    T = Atom[lift(a) for a in p.targets]
    G = Atom[x for x in (lift(a) for a in p.given) if x !== nothing]
    return T, G
end

# Names for the values of the free (valueless) atoms of a query: the variable's
# own name the first time, then primed (Y, Y′, …), in order of appearance.
function _cf_valuenames(atoms)
    used = Dict{RV,Int}()
    out = Dict{Atom,RV}()
    for a in atoms
        v = a isa Counterfactual ? (a.hasvalue ? nothing : a.rv) : (a isa RV && !isevent(a) ? a : nothing)
        (v === nothing || haskey(out, a)) && continue
        k = get(used, v, 0)
        used[v] = k + 1
        out[a] = k == 0 ? v : RV(Symbol(v.name, "′"^k), v.kind)
    end
    return out
end

function _cf_eval(p::Prob, bn::BayesNet)
    T, G = _cf_atoms(p)
    # every world, the worlds of nested potential outcomes included
    worlds = Vector{Atom}[Atom[]]
    addworld!(sub) = begin
        sub in worlds || push!(worlds, sub)
        for e in sub
            e isa Counterfactual && addworld!(e.sub)
        end
    end
    for a in vcat(T, G)
        a isa Counterfactual && addworld!(a.sub)
    end
    isroot(v) = isempty(bn.dag.parents[v])
    # the node that carries `v` in world `w`
    function wnode(v::RV, w::Vector{Atom})
        i = findfirst(e -> _subrv(e) == v, w)
        i === nothing && return isroot(v) ? v : _worldname(v, w)
        e = w[i]
        return e isa Event ? _worldname(v, w) : wnode(e.rv, e.sub)
    end
    specs = Pair[]
    doms = Dict{RV,Any}()
    for w in worlds, v in topological_order(bn.dag)
        i = findfirst(e -> _subrv(e) == v, w)
        if i !== nothing
            w[i] isa Event || continue          # nested: no node of its own
            d = domain(bn, v)
            k = findfirst(x -> isequal(x, w[i].value), d)
            k === nothing && throw(ArgumentError("$(repr(w[i].value)) is not a value of $(v.name)"))
            T0 = zeros(length(d)); T0[k] = 1.0
            push!(specs, _worldname(v, w) => T0)
            doms[_worldname(v, w)] = d
        elseif isroot(v)
            isempty(w) && (push!(specs, v => copy(bn.cpts[v].values)); doms[v] = domain(bn, v))
        else
            vals = bn.cpts[v].values
            all(x -> x == 0 || x == 1, vals) || throw(ArgumentError(
                "counterfactuals need a structural causal model: $(v.name) is not a deterministic " *
                "function of its parents (give it an exogenous noise parent, or build the model with rand_scm)"))
            name = _worldname(v, w)
            push!(specs, Given(Atom[name], Atom[(wnode(q, w) for q in bn.dag.parents[v])...]) => copy(vals))
            doms[name] = domain(bn, v)
        end
    end
    pw = BayesNet(specs...; domains=doms)
    names = _cf_valuenames(vcat(T, G))
    back = Dict{RV,RV}()
    function cv(a::Atom)
        a isa Counterfactual || return a
        n = wnode(a.rv, a.sub)
        a.hasvalue && return Event(n, a.value)
        back[n] = names[a]
        return n
    end
    q = Prob(Atom[cv(a) for a in T], Atom[cv(a) for a in G], p.name)
    for (a, nm) in names
        a isa RV && (back[a] = nm)
    end
    f = _eval(q, pw)
    isempty(f.vars) && return f
    return Factor(RV[get(back, v, v) for v in f.vars], f.domains, f.values, f.label)
end

# ---------------------------------------------------------------------------
# Identification: make-cg, ID*, IDC*
# ---------------------------------------------------------------------------

# A value: concrete, or the value of a (summed) copy of a variable.
struct _Val
    v::Any
    free::Bool
end
Base.:(==)(a::_Val, b::_Val) = a.free == b.free && isequal(a.v, b.v)
Base.hash(a::_Val, h::UInt) = hash(a.v, hash(a.free, h))

# A counterfactual event: var_{sub} = val
struct _CFE
    var::RV
    sub::Dict{RV,_Val}
    val::_Val
end

const _Node = Tuple{RV,Int}

mutable struct _CFCtx
    G::ADMG
    order::Vector{RV}
    name::Symbol
    copies::Dict{RV,Int}
    query::PExpr
    reserved::Dict{RV,Int}       # names taken by the query's own free values
end
function _fresh!(ctx::_CFCtx, v::RV)
    k = get(ctx.copies, v, 0)
    ctx.copies[v] = k + 1
    return k == 0 ? v : RV(Symbol(v.name, "′"^k), v.kind)
end

struct _CG
    nodes::Vector{_Node}                 # non-constant ancestors of γ′, topological
    parents::Dict{_Node,Vector{_Node}}
    consts::Dict{_Node,_Val}             # constant (intervened) nodes and their values
    values::Dict{_Node,_Val}             # γ′: node values
    nodeof::Vector{_Node}                # the node of each input event (or a constant node)
end

function _make_cg(ctx::_CFCtx, γ::Vector{_CFE})
    G, order = ctx.G, ctx.order
    worlds = Dict{RV,_Val}[Dict{RV,_Val}()]
    for e in γ
        any(w -> w == e.sub, worlds) || push!(worlds, e.sub)
    end
    windex(sub) = findfirst(w -> w == sub, worlds)
    rep = Dict{_Node,_Node}()
    consts = Dict{_Node,_Val}()
    constnode = Dict{Tuple{RV,_Val},_Node}()     # one constant node per variable and value
    values = Dict{_Node,_Val}()
    pa = Dict{_Node,Vector{_Node}}()
    # Parents match when they are the same node or are known to have the same
    # value (fixed by intervention or observed in γ): Lemma 24 of Shpitser–Pearl.
    pkey(q::_Node) = haskey(consts, q) ? (:val, consts[q]) : haskey(values, q) ? (:val, values[q]) : (:node, q)
    for v in order
        for (wi, w) in enumerate(worlds)
            n = (v, wi)
            if haskey(w, v)
                rep[n] = get!(constnode, (v, w[v])) do
                    c = (v, -(length(constnode) + 1))
                    consts[c] = w[v]
                    c
                end
                continue
            end
            ps = _Node[rep[(p, wi)] for p in parents(G, v)]
            key = [pkey(q) for q in ps]
            merged = false
            for wj in 1:wi-1
                m = rep[(v, wj)]
                haskey(consts, m) && continue
                if [pkey(q) for q in pa[m]] == key
                    rep[n] = m
                    merged = true
                    break
                end
            end
            if !merged
                rep[n] = n
                pa[n] = ps
            end
        end
        # the events on v fix the values of its nodes
        for e in γ
            e.var == v || continue
            m = rep[(v, windex(e.sub))]
            if haskey(consts, m)
                consts[m] == e.val || return nothing
            elseif haskey(values, m)
                values[m] == e.val || return nothing
            else
                values[m] = e.val
            end
        end
    end
    nodeof = _Node[rep[(e.var, windex(e.sub))] for e in γ]
    # ancestors of the events' nodes
    keep = Set{_Node}()
    stack = _Node[n for n in nodeof if !haskey(consts, n)]
    while !isempty(stack)
        n = pop!(stack)
        n in keep && continue
        push!(keep, n)
        for q in pa[n]
            haskey(consts, q) || push!(stack, q)
        end
    end
    nodes = _Node[]
    for v in order, wi in eachindex(worlds)
        m = rep[(v, wi)]
        m in keep && !(m in nodes) && push!(nodes, m)
    end
    return _CG(nodes, Dict(n => pa[n] for n in nodes), consts,
               Dict(n => x for (n, x) in values if n in keep), nodeof)
end

# Bidirected connections in the counterfactual graph: copies of one variable
# share its exogenous noise, and hidden confounders are shared across worlds.
_cg_linked(G::ADMG, a::_Node, b::_Node) =
    a != b && (a[1] == b[1] || (a[1] < b[1] ? (a[1], b[1]) : (b[1], a[1])) in G.bidirected)

function _cg_districts(G::ADMG, ns::Vector{_Node})
    seen = Set{_Node}()
    out = Vector{_Node}[]
    for n in ns
        n in seen && continue
        comp = _Node[]
        stack = _Node[n]
        push!(seen, n)
        while !isempty(stack)
            x = pop!(stack)
            push!(comp, x)
            for y in ns
                y in seen || !_cg_linked(G, x, y) || (push!(seen, y); push!(stack, y))
            end
        end
        push!(out, _Node[x for x in ns if x in comp])
    end
    return out
end

# The counterfactual graph as a DAG (nodes as RVs, latent nodes for the
# bidirected connections), for m-separation in IDC*.
function _cg_dag(G::ADMG, cg::_CG)
    nm(n::_Node) = RV(Symbol(n[1].name, "@", n[2]))
    ns = RV[nm(n) for n in cg.nodes]
    pa = Dict{RV,Vector{RV}}(nm(n) => RV[nm(q) for q in cg.parents[n] if !haskey(cg.consts, q)] for n in cg.nodes)
    k = 0
    for i in eachindex(cg.nodes), j in i+1:length(cg.nodes)
        _cg_linked(G, cg.nodes[i], cg.nodes[j]) || continue
        k += 1
        u = RV(Symbol("U@", k))
        push!(ns, u)
        pa[u] = RV[]
        push!(pa[nm(cg.nodes[i])], u)
        push!(pa[nm(cg.nodes[j])], u)
    end
    return DAG(ns, pa), nm
end

function _cf_fail(ctx::_CFCtx, S)
    vs = unique(RV[n[1] for n in S])
    throw(NotIdentifiable(ctx.query, (vs, vs)))
end

function _idstar(ctx::_CFCtx, γ::Vector{_CFE})
    isempty(γ) && return Literal(1)
    keep = _CFE[]
    for e in γ
        if haskey(e.sub, e.var)                # x_{x} = x always; x_{x} = x′ never
            e.sub[e.var] == e.val || return Literal(0)
            continue
        end
        push!(keep, e)
    end
    isempty(keep) && return Literal(1)
    cg = _make_cg(ctx, keep)
    cg === nothing && return Literal(0)
    free = Dict{_Node,_Val}(n => _Val(_fresh!(ctx, n[1]), true) for n in cg.nodes if !haskey(cg.values, n))
    val(n) = haskey(cg.consts, n) ? cg.consts[n] : haskey(cg.values, n) ? cg.values[n] : free[n]
    summed = RV[free[n].v for n in cg.nodes if haskey(free, n)]
    # The counterfactual graph factorises over its districts, Q[S] = P(S | do(pa(S) ∖ S));
    # each Q[S] is an interventional distribution of the original model when
    # no variable needs two different values in it (line 7 of ID*).
    parts = PExpr[_district_term(ctx, cg, S, val) for S in _cg_districts(ctx.G, cg.nodes)]
    return marg(summed, mul(parts...))
end

function _district_term(ctx::_CFCtx, cg::_CG, S::Vector{_Node}, val)
    subs = Dict{RV,_Val}()
    for n in S, q in cg.parents[n]
        q in S && continue
        x = val(q)
        haskey(subs, q[1]) && subs[q[1]] != x && _cf_fail(ctx, S)
        subs[q[1]] = x
    end
    vals = Dict{RV,_Val}()
    for n in S
        x = val(n)
        haskey(vals, n[1]) && vals[n[1]] != x && _cf_fail(ctx, S)
        vals[n[1]] = x
    end
    for (v, x) in collect(vals)
        haskey(subs, v) || continue
        subs[v] == x || _cf_fail(ctx, S)
        delete!(vals, v)
    end
    return _cf_term(ctx, vals, subs)
end

# P(vals | do(subs)) in observational terms (free copies renamed back after).
function _cf_term(ctx::_CFCtx, vals, subs)
    targets = Atom[]
    given = Atom[]
    back = Dict{Atom,Atom}()
    for v in _inorder(ctx.order, keys(vals))
        x = vals[v]
        x.free ? (push!(targets, v); x.v == v || (back[v] = x.v)) : push!(targets, Event(v, x.v))
    end
    isempty(targets) && return Literal(1)
    for v in _inorder(ctx.order, keys(subs))
        x = subs[v]
        x.free ? (push!(given, Intervention(v)); x.v == v || (back[v] = x.v)) :
                 push!(given, Intervention(Event(v, x.v)))
    end
    e = identify(Prob(targets, given, ctx.name), ctx.G)
    return isempty(back) ? e : _subs(e, back)
end

_reset!(ctx::_CFCtx) = (empty!(ctx.copies); merge!(ctx.copies, ctx.reserved); ctx)

function _idcstar(ctx::_CFCtx, γ::Vector{_CFE}, δ::Vector{_CFE})
    # numerator and denominator are separate expressions, so each names its
    # summed copies afresh (W, W′, …), after the names the query itself uses
    _reset!(ctx)
    isempty(δ) && return _idstar(ctx, γ)
    den = _idstar(ctx, δ)
    _reset!(ctx)
    den == Literal(0) && throw(ArgumentError("the conditioning event of $(ctx.query) is impossible"))
    cg = _make_cg(ctx, vcat(γ, δ))
    cg === nothing && return Literal(0)
    dag, nm = _cg_dag(ctx.G, cg)
    nγ = cg.nodeof[1:length(γ)]
    nδ = cg.nodeof[length(γ)+1:end]
    for (k, d) in enumerate(δ)
        nd = nδ[k]
        haskey(cg.consts, nd) && continue
        others = RV[nm(nδ[j]) for j in eachindex(δ) if j != k && !haskey(cg.consts, nδ[j])]
        targets = RV[nm(n) for n in nγ if !haskey(cg.consts, n)]
        isempty(targets) && continue
        nm(nd) in targets && continue
        if d_separated(remove_outgoing(dag, nm(nd)), targets, nm(nd), others)
            # condition becomes intervention: γ_{d}
            γ2 = _CFE[haskey(e.sub, d.var) ? e : _CFE(e.var, merge(e.sub, Dict(d.var => d.val)), e.val) for e in γ]
            return _idcstar(ctx, γ2, _CFE[δ[j] for j in eachindex(δ) if j != k])
        end
    end
    return divide(_idstar(ctx, vcat(γ, δ)), den)
end

# An atom of the query as a counterfactual event. Free atoms (no value) take
# their value names; nested potential outcomes are unnested (Correa, Lee &
# Bareinboim 2021): Y_{x, M_{x′}} = y is Y_{x, m} = y together with
# M_{x′} = m, summed over m. The extra events and summed values are collected.
function _tocfe(a::Atom, names, ctx::_CFCtx, extra::Vector{_CFE}, summed::Vector{RV})
    if a isa Counterfactual
        val = a.hasvalue ? _Val(a.value, false) : _Val(names[a], true)
        sub = Dict{RV,_Val}()
        for e in a.sub
            if e isa Event
                sub[e.rv] = _Val(e.value, false)
            else
                m = _fresh!(ctx, e.rv)
                ctx.reserved[e.rv] = ctx.copies[e.rv]
                push!(summed, m)
                push!(extra, _tocfe(Counterfactual(e.rv, e.sub, m, true), names, ctx, extra, summed) |> x ->
                      _CFE(x.var, x.sub, _Val(m, true)))
                sub[e.rv] = _Val(m, true)
            end
        end
        return _CFE(a.rv, sub, val)
    elseif a isa Event
        return _CFE(a.rv, Dict{RV,_Val}(), _Val(a.value, false))
    elseif a isa RV && !isevent(a)
        return _CFE(a, Dict{RV,_Val}(), _Val(names[a], true))
    end
    throw(ArgumentError("counterfactual queries take variables, events X(1) and potential outcomes; got $(_astr(a))"))
end

_cfvars(a::Counterfactual) = vcat(RV[a.rv], (e isa Event ? RV[e.rv] : _cfvars(e) for e in a.sub)...)
_cfvars(a::Atom) = rvs(a)

"""
    identify(P(Y(Do(X(x)))(y) | X(x′)), g::ADMG; trace = false)

Counterfactual queries (conjunctions of potential outcomes and ordinary
events, optionally conditioned) are identified with the ID* / IDC*
algorithms of Shpitser and Pearl (2007): the query is drawn as a counterfactual
graph (make-cg, which merges the copies of a variable that must be equal), split
into districts, and each district identified by ID. Throws
[`NotIdentifiable`](@ref) when the graph does not determine the answer (e.g.
the probability of necessity and sufficiency `P(Y(Do(X(1)))(1), Y(Do(X(0)))(0))`).

Potential outcomes without a value (`P(Y(Do(X(1))), X)`) make the answer a
function of their values, named after the variable (`Y`, then `Y′`, … when a
variable appears more than once). Nested potential outcomes such as
`Y(Do(X(1)), M(Do(X(0))))` (`Y_{x, M_{x′}}`, for natural direct and indirect
effects, see [`mediation`](@ref)) are unnested first:
`P(Y_{x, M_{x′}} = y) = ∑_m P(Y_{x, m} = y, M_{x′} = m)`.
"""
function _identify_cf(p::Prob, g::ADMG; trace::Bool=false)
    T, G = _cf_atoms(p)
    for a in vcat(T, G), v in _cfvars(a)
        v in g.dag.nodes || throw(ArgumentError("$(v.name) is not a node of the graph"))
    end
    names = _cf_valuenames(vcat(T, G))
    ctx = _CFCtx(g, topological_order(g), p.name, Dict{RV,Int}(), p, Dict{RV,Int}())
    for nm in values(names)
        b = _unprimed(nm)
        v = b === nothing ? nm : b
        while (u = _unprimed(v)) !== nothing
            v = u
        end
        ctx.reserved[v] = get(ctx.reserved, v, 0) + 1
    end
    _reset!(ctx)
    γx, γs, δx, δs = _CFE[], RV[], _CFE[], RV[]
    γ = _CFE[_tocfe(a, names, ctx, γx, γs) for a in T]
    δ = _CFE[_tocfe(a, names, ctx, δx, δs) for a in G]
    e = if isempty(δs)
        marg(γs, _idcstar(ctx, vcat(γ, γx), δ))
    else
        _reset!(ctx)
        num = marg(vcat(γs, δs), _idstar(ctx, vcat(γ, γx, δ, δx)))
        _reset!(ctx)
        divide(num, marg(δs, _idstar(ctx, vcat(δ, δx))))
    end
    e = _freshen(e, Set{RV}(values(names)))
    d = derivation(p)
    _push!(d, e, isempty(G) ? "ID* algorithm (Shpitser–Pearl)" : "IDC* algorithm (Shpitser–Pearl)")
    apply!(d, x -> simplify(x; trace=true))
    return trace ? d : last(d)
end

# Plates: indexed variables and products over an index.
#
#   @index i 1:n          an index i running over 1..n (bounds may be symbols)
#   y[i], x[t-1], x[1:t]  indexed variables (one element, a shifted element, a range)
#   y[3]                  a concrete element: the ordinary variable y_3
#   ∏(i, expr)            a product over the plate
#
# Rules: `sequence` (the chain rule over a sequence), `peel` (split off the
# last element), and simplify rules for `iid` and `markov` facts. `unroll`
# replaces symbolic indices by numbers, giving ordinary expressions that
# `compute` and `verify` can check.

"""
    Index(name, lo, hi)
    @index i 1:n

An index running from `lo` to `hi`. Bounds may be integers, symbols (`:n`),
or other indices (`1:t`). `i + 1` and `i - 1` are shifted indices.
"""
struct Index
    name::Symbol
    lo::Any
    hi::Any
end
"""
    IndexShift(index, k)

`index + k`, as in `x[t-1]`.
"""
struct IndexShift
    index::Index
    k::Int
end
"""
    IndexRange(lo, hi)

`lo:hi` with a symbolic end, as in `x[1:t]` or `x[1:t-1]`.
"""
struct IndexRange
    lo::Any
    hi::Any
end
const _IndexLike = Union{Index,IndexShift}
const _AnyIndex = Union{Index,IndexShift,IndexRange}

Base.:(==)(a::Index, b::Index) = a.name === b.name && isequal(a.lo, b.lo) && isequal(a.hi, b.hi)
Base.hash(a::Index, h::UInt) = hash(a.hi, hash(a.lo, hash(a.name, hash(:Index, h))))
Base.:(==)(a::IndexShift, b::IndexShift) = a.index == b.index && a.k == b.k
Base.hash(a::IndexShift, h::UInt) = hash(a.k, hash(a.index, hash(:IndexShift, h)))
Base.:(==)(a::IndexRange, b::IndexRange) = isequal(a.lo, b.lo) && isequal(a.hi, b.hi)
Base.hash(a::IndexRange, h::UInt) = hash(a.hi, hash(a.lo, hash(:IndexRange, h)))
Base.isequal(a::_AnyIndex, b::_AnyIndex) = a == b

"""
    whole(i)

The whole range of an index, `i.lo:i.hi`, for use as `y[whole(i)]` (shown
`y[1:n]`) when the end is symbolic.
"""
whole(i::Index) = IndexRange(i.lo, i.hi)

Base.:+(i::Index, k::Integer) = k == 0 ? i : IndexShift(i, k)
Base.:-(i::Index, k::Integer) = i + (-k)
Base.:+(s::IndexShift, k::Integer) = s.index + (s.k + k)
Base.:-(s::IndexShift, k::Integer) = s + (-k)
Base.:(:)(a::Integer, b::_IndexLike) = IndexRange(a, b)
Base.:(:)(a::_IndexLike, b::Union{Integer,_IndexLike}) = IndexRange(a, b)

"""
    @index i 1:n
    @index i ∈ 1:n
    @index t 1:T s 1:S

Declare indices, binding each name in the calling scope. Symbols in the bounds
(`n`, `T`) stay symbolic; numbers stay numbers.
"""
macro index(args...)
    pairs = Tuple{Symbol,Any}[]
    a = collect(args)
    while !isempty(a)
        x = popfirst!(a)
        if x isa Expr && x.head === :call && x.args[1] in (:∈, :in) && x.args[2] isa Symbol
            push!(pairs, (x.args[2], x.args[3]))
        elseif x isa Symbol && !isempty(a)
            push!(pairs, (x, popfirst!(a)))
        else
            error("@index: write `@index i 1:n` or `@index i ∈ 1:n`")
        end
    end
    out = Expr(:block)
    for (name, r) in pairs
        (r isa Expr && r.head === :call && r.args[1] === :(:) && length(r.args) == 3) ||
            error("@index: the range of $name must be `lo:hi`, got `$r`")
        push!(out.args, :($(esc(name)) = $Index($(QuoteNode(name)), $(_boundq(r.args[2])), $(_boundq(r.args[3])))))
    end
    push!(out.args, Expr(:tuple, (esc(n) for (n, _) in pairs)...))
    return out
end
# A bound: numbers as they are; a bound Index variable by value; other names symbolic.
_boundq(x::Integer) = x
_boundq(s::Symbol) = esc(:((@isdefined($s) && $s isa $Index) ? $s : $(QuoteNode(s))))
_boundq(x) = QuoteNode(x)

"""
    IndexedRV(base, index)

An element (`y[i]`, `x[t-1]`) or a range (`x[1:t]`) of an indexed random
variable, with a symbolic index. Built by indexing an [`RV`](@ref) with an
[`Index`](@ref). With a number, `y[3]` is the ordinary variable `y_3`.
"""
struct IndexedRV <: Atom
    base::RV
    index::_AnyIndex
end
Base.:(==)(a::IndexedRV, b::IndexedRV) = a.base == b.base && a.index == b.index
Base.hash(a::IndexedRV, h::UInt) = hash(a.index, hash(a.base, hash(:IndexedRV, h)))

Base.getindex(x::RV, k::Integer) = RV(Symbol(x.name, "_", k), x.kind)
Base.getindex(x::RV, r::AbstractUnitRange{<:Integer}) = RV[x[k] for k in r]
Base.getindex(x::RV, i::_AnyIndex) = IndexedRV(x, i)

# Index text
_istr(x::Integer) = string(x)
_istr(s::Symbol) = string(s)
_istr(i::Index) = string(i.name)
_istr(s::IndexShift) = string(s.index.name, s.k < 0 ? "-" : "+", abs(s.k))
_istr(r::IndexRange) = string(_istr(r.lo), ":", _istr(r.hi))
_istr(x) = string(x)

# A stand-in variable for graph/variable bookkeeping (`y_i`, `x_t-1`, `x_1:t`).
_flatrv(a::IndexedRV) = RV(Symbol(a.base.name, "_", _istr(a.index)), a.base.kind)
rv(a::IndexedRV) = _flatrv(a)
rvs(a::IndexedRV) = RV[_flatrv(a)]
freerv(a::IndexedRV) = isevent(a.base) ? nothing : _flatrv(a)
iseventlike(a::IndexedRV) = isevent(a.base)

_astr(a::IndexedRV, inner=false) = string(a.base.name, "[", _istr(a.index), "]")
_lxidx(x::Integer) = string(x)
_lxidx(s::Symbol) = _lx(s)
_lxidx(i::Index) = _lx(i.name)
_lxidx(s::IndexShift) = string(_lx(s.index.name), s.k < 0 ? "-" : "+", abs(s.k))
_lxidx(r::IndexRange) = string(_lxidx(r.lo), ":", _lxidx(r.hi))
_lxidx(x) = _lxcode(x)
_lxatom(a::IndexedRV, inner=false) = string(_lx(a.base.name), "_{", _lxidx(a.index), "}")
Base.show(io::IO, i::Index) = print(io, i.name, " ∈ ", _istr(i.lo), ":", _istr(i.hi))
Base.show(io::IO, i::IndexShift) = print(io, _istr(i))
Base.show(io::IO, i::IndexRange) = print(io, _istr(i))

# Does an index expression involve the index `i`?
_uses(x, ::Index) = false
_uses(j::Index, i::Index) = j.name === i.name
_uses(s::IndexShift, i::Index) = _uses(s.index, i)
_uses(r::IndexRange, i::Index) = _uses(r.lo, i) || _uses(r.hi, i)
_uses(a::IndexedRV, i::Index) = _uses(a.index, i)
_uses(::Atom, ::Index) = false

# ---------------------------------------------------------------------------
# Products over a plate
# ---------------------------------------------------------------------------

"""
    PlateProduct(index, body)
    ∏(i, body)

The product of `body` over the index `i`: `∏_{i=lo}^{hi} body`.
"""
struct PlateProduct <: PExpr
    index::Index
    body::PExpr
end
∏(i::Index, body::PExpr) = body isa Literal && body.value == 1 ? body : PlateProduct(i, body)
const plate = ∏

_atomic(p::PlateProduct) = _atomic(p.body)     # ∏ᵢ P(⋯) needs no brackets in a product
Base.:(==)(a::PlateProduct, b::PlateProduct) = a.index == b.index && a.body == b.body
Base.hash(a::PlateProduct, h::UInt) = hash(a.body, hash(a.index, hash(:PlateProduct, h)))
children(p::PlateProduct) = PExpr[p.body]
rebuild(p::PlateProduct, cs) = ∏(p.index, cs[1])
_subs(p::PlateProduct, d) = ∏(p.index, _subs(p.body, d))

_atoms_in(p::Prob) = vcat(p.targets, p.given)
_atoms_in(e::PExpr) = reduce(vcat, (_atoms_in(c) for c in children(e)); init=Atom[])
_freevars(p::PlateProduct) = Set{RV}(v for a in _atoms_in(p.body) if !_uses(a, p.index)
                                     for v in (freerv(a isa Intervention ? a.target : a),) if v !== nothing)
_allvars(p::PlateProduct) = _allvars(p.body)

function Base.show(io::IO, p::PlateProduct)
    print(io, "∏_{", p.index.name, "=", _istr(p.index.lo), "}^{", _istr(p.index.hi), "} ")
    p.body isa Union{Sum,Quotient} ? _show_paren(io, p.body) : show(io, p.body)
end
_latex(p::PlateProduct) = string("\\prod_{", _lx(p.index.name), "=", _lxidx(p.index.lo), "}^{", _lxidx(p.index.hi), "} ",
                                 p.body isa Union{Sum,Quotient} ? _lxparen(p.body) : _latex(p.body))

_eval(p::PlateProduct, ::_Nets) =
    throw(ArgumentError("a product over a symbolic index can't be computed; `unroll` it first (unroll(expr, :$(p.index.hi) => 3))"))

# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------

function _range_target(p::Prob)
    k = findfirst(a -> a isa IndexedRV && a.index isa IndexRange, p.targets)
    k === nothing && throw(ArgumentError("$(p) has no indexed range among its targets (like x[1:n])"))
    return k, p.targets[k]
end

"""
    sequence(P(x[1:n] | C), i) -> ∏_{i=1}^{n} P(x[i] | x[1:i-1], C)

The chain rule over a sequence, as a product over the index `i` (whose name
is used; its bounds come from the range).
"""
function sequence(p::Prob, i::Index)
    k, a = _range_target(p)
    j = Index(i.name, a.index.lo, a.index.hi)
    others = Atom[t for (m, t) in enumerate(p.targets) if m != k]
    isempty(others) || throw(ArgumentError("sequence takes one range as the target; condition on the rest first"))
    past = IndexedRV(a.base, IndexRange(a.index.lo, j - 1))
    return ∏(j, _like(p, Atom[IndexedRV(a.base, j)], vcat(past, p.given)))
end

"""
    peel(P(x[1:t] | C)) -> P(x[t] | x[1:t-1], C) P(x[1:t-1] | C)

Split the last element off a sequence (one step of the chain rule), as in a
filtering or SMC recursion.
"""
function peel(p::Prob)
    k, a = _range_target(p)
    hi = a.index.hi
    hi isa _IndexLike || hi isa Integer ||
        throw(ArgumentError("the end of the range must be an index or a number to peel it"))
    last_ = hi isa Integer ? a.base[hi] : IndexedRV(a.base, hi)
    prev = hi isa Integer ? (hi - 1 >= a.index.lo ? a.base[a.index.lo:hi-1] : RV[]) :
           Atom[IndexedRV(a.base, IndexRange(a.index.lo, hi - 1))]
    others = Atom[t for (m, t) in enumerate(p.targets) if m != k]
    return mul(_like(p, vcat(Atom[last_], others), vcat(prev, p.given)),
               isempty(prev) && isempty(others) ? Literal(1) : _like(p, vcat(prev, others), p.given))
end

# ---------------------------------------------------------------------------
# iid and Markov facts
# ---------------------------------------------------------------------------

_elementof(a, base::RV) = a isa IndexedRV && a.base == base

function rule_iid(p::Prob, ctx::SimplifyContext)
    isempty(ctx.facts.iid) && return nothing
    for f in ctx.facts.iid
        all(g -> g in p.given, f.given) || continue
        # P(y[i] | y[⋯], θ) = P(y[i] | θ)
        if length(p.targets) == 1 && _elementof(p.targets[1], f.base) && !(p.targets[1].index isa IndexRange)
            drop = [g for g in p.given if _elementof(g, f.base) || (g isa RV && startswith(string(g.name), string(f.base.name, "_")))]
            isempty(drop) && continue
            return (_like(p, p.targets, Atom[g for g in p.given if !(g in drop)]),
                    "$(f.base.name)[i] are iid given $(_atomlist(f.given)) (assumed)")
        end
        # P(y[lo:hi] | θ) = ∏ᵢ P(y[i] | θ)
        if length(p.targets) == 1 && _elementof(p.targets[1], f.base) && p.targets[1].index isa IndexRange &&
           all(g -> g in f.given, p.given)
            r = p.targets[1].index
            i = Index(:i, r.lo, r.hi)
            return (∏(i, _like(p, Atom[IndexedRV(f.base, i)], p.given)),
                    "$(f.base.name)[i] are iid given $(_atomlist(f.given)) (assumed): the joint is a product")
        end
    end
    return nothing
end
rule_iid(::PExpr, ::SimplifyContext) = nothing

function rule_markov(p::Prob, ctx::SimplifyContext)
    isempty(ctx.facts.markov) && return nothing
    for m in ctx.facts.markov, t in p.targets
        _elementof(t, m.base) && t.index isa _IndexLike || continue
        prev = t.index - 1
        for g in p.given
            _elementof(g, m.base) && g.index isa IndexRange && g.index.hi == prev || continue
            g.index.lo == prev && continue
            newgiven = Atom[x == g ? IndexedRV(m.base, prev) : x for x in p.given]
            return (_like(p, p.targets, newgiven), "Markov property of $(m.base.name)")
        end
    end
    return nothing
end
rule_markov(::PExpr, ::SimplifyContext) = nothing

push!(_RULES, rule_iid, rule_markov)

# ---------------------------------------------------------------------------
# Unrolling
# ---------------------------------------------------------------------------

"""
    unroll(expr, bindings...)

Replace symbolic indices and bounds by numbers: `unroll(e, :n => 3)` expands
every `∏_{i=1}^{n}` into three factors and `y[1:n]` into `y_1, y_2, y_3`;
`unroll(e, t => 2)` fixes the index `t`. The result is an ordinary expression
that [`compute`](@ref) and [`verify`](@ref) can evaluate on a network whose
variables are named `y_1`, `y_2`, … (`y[1]`, `y[2]`, …).
"""
function unroll(e::PExpr, bindings::Pair...)
    env = Dict{Symbol,Int}()
    for (k, v) in bindings
        env[k isa Index ? k.name : Symbol(k)] = v
    end
    return _unroll(e, env)
end

_ival(x::Integer, env) = x
_ival(s::Symbol, env) = haskey(env, s) ? env[s] : throw(ArgumentError("no value for $(s); pass `:$(s) => …` to unroll"))
_ival(i::Index, env) = _ival(i.name, env)
_ival(s::IndexShift, env) = _ival(s.index, env) + s.k
# `T - 1`, `T + 2`: a symbolic bound shifted by a constant (from peeling)
function _ival(x::Expr, env)
    if x.head === :call && x.args[1] in (:+, :-) && length(x.args) == 3 && x.args[3] isa Integer
        v = _ival(x.args[2], env)
        return x.args[1] === :+ ? v + x.args[3] : v - x.args[3]
    end
    throw(ArgumentError("cannot unroll the bound $(x)"))
end
_ival(x, env) = throw(ArgumentError("cannot unroll the bound $(x)"))

_unroll_atom(a::Atom, env) = Atom[a]
function _unroll_atom(a::IndexedRV, env)
    if a.index isa IndexRange
        return Atom[a.base[k] for k in _ival(a.index.lo, env):_ival(a.index.hi, env)]
    end
    return Atom[a.base[_ival(a.index, env)]]
end
_unroll_atom(c::Complement, env) = Atom[_complement(only(_unroll_atom(c.event, env)))]
_unroll_atom(i::Intervention, env) = Atom[Intervention(t) for t in _unroll_atom(i.target, env)]

_unroll(l::Literal, env) = l
_unroll(p::Prob, env) = _mkprob(reduce(vcat, (_unroll_atom(a, env) for a in p.targets); init=Atom[]),
                                reduce(vcat, (_unroll_atom(a, env) for a in p.given); init=Atom[]); name=p.name)
function _unroll(p::PlateProduct, env)
    lo, hi = _ival(p.index.lo, env), _ival(p.index.hi, env)
    return mul((_unroll(p.body, merge(env, Dict(p.index.name => k))) for k in lo:hi)...)
end
_unroll(e::PExpr, env) = rebuild(e, PExpr[_unroll(c, env) for c in children(e)])

# ---------------------------------------------------------------------------
# Generative models with plates
# ---------------------------------------------------------------------------

# The variable a reference in a `~` right-hand side denotes, inside loops.
function _plate_atom(ex, rvmap, loops)
    if ex isa Symbol
        return get(rvmap, ex, nothing)
    elseif ex isa Expr && ex.head === :ref && ex.args[1] isa Symbol && length(ex.args) == 2
        base = get(rvmap, ex.args[1], nothing)
        base === nothing && return nothing
        idx = _plate_index(ex.args[2], loops)
        return idx === nothing ? base : (idx isa Integer ? base[idx] : IndexedRV(base, idx))
    end
    return nothing
end

function _plate_index(ix, loops)
    ix isa Integer && return ix
    for (i, r) in loops
        I = _loop_index(i, r)
        ix === i && return I
        if ix isa Expr && ix.head === :call && ix.args[1] in (:+, :-) && length(ix.args) == 3 &&
           ix.args[2] === i && ix.args[3] isa Integer
            return ix.args[1] === :+ ? I + ix.args[3] : I - ix.args[3]
        end
    end
    return nothing
end

function _loop_index(i::Symbol, r)
    if r isa Expr && r.head === :call && r.args[1] === :(:) && length(r.args) == 3
        return Index(i, r.args[2], r.args[3])
    end
    return Index(i, 1, r)     # e.g. eachindex(y): shown as the range expression
end

# The variables each `~` line itself depends on (not the node's union over all
# its lines: `x[1] ~ Normal(0, 1)` has none even if `x[t] ~ Normal(ρ x[t-1], 1)`
# depends on ρ), by replaying the model's statements in order.
function _lines_parents(gm::GenerativeModel)
    stoch = Set(v.name for v in vcat(gm.latent, gm.observed))
    args = Set(gm.args)
    det = Dict{Symbol,Set{Symbol}}()
    out = Dict{Int,Vector{Symbol}}()
    for (k, l) in enumerate(gm.lines)
        loopvars = Set(first.(l.loops))
        reads = _readsyms(l.rhs)
        if l.kind === :tilde && l.lhs isa Expr && l.lhs.head === :ref
            foreach(i -> _readsyms(i, reads), l.lhs.args[2:end])
        end
        names = Symbol[]
        for s in reads
            (s in loopvars || (l.kind === :tilde && s === l.node)) && continue
            add = s in stoch ? (s,) : haskey(det, s) ? det[s] : s in args ? (s,) : ()
            for a in add
                a in names || push!(names, a)
            end
        end
        if l.kind === :assign
            l.node in stoch || (det[l.node] = union(get(det, l.node, Set{Symbol}()), names))
        else
            out[k] = names
        end
    end
    return out
end

# Parent atoms of one `~` line: indexed references where the line indexes them,
# then the line's other parents.
function _line_parents(l::ModelLine, names::Vector{Symbol}, rvmap)
    out = Atom[]
    seen = Set{Symbol}()
    function walk(ex)
        if ex isa Expr && ex.head === :ref && ex.args[1] isa Symbol &&
           (ex.args[1] === l.node || ex.args[1] in names)
            a = _plate_atom(ex, rvmap, l.loops)
            if a !== nothing
                a in out || push!(out, a)
                push!(seen, ex.args[1])
                return
            end
        end
        ex isa Expr && foreach(walk, ex.head === :call ? ex.args[2:end] : ex.args)
    end
    walk(l.rhs)
    for s in names         # the rest (unindexed, or through deterministic assignments)
        s in seen || !haskey(rvmap, s) || push!(out, rvmap[s])
    end
    return out
end

function _plate_factorize(gm::GenerativeModel, name::Symbol)
    rvmap = Dict(v.name => v for v in gm.dag.nodes)
    lp = _lines_parents(gm)
    fs = PExpr[]
    for v in topological_order(gm.dag)
        v in gm.inputs && continue
        ks = [k for (k, l) in enumerate(gm.lines) if l.kind === :tilde && l.node === v.name]
        lines = gm.lines[ks]
        plain = all(l -> l.lhs isa Symbol && isempty(l.loops), lines) || any(l -> l.dot, lines)
        if plain
            push!(fs, Prob(Atom[v], Atom[gm.dag.parents[v]...], name))
            continue
        end
        for (k, l) in zip(ks, lines)
            target = _plate_atom(l.lhs, rvmap, l.loops)
            target === nothing && (target = v)
            pars = _line_parents(l, lp[k], rvmap)
            f = Prob(Atom[target], Atom[x for x in pars if x != target], name)
            for (i, r) in reverse(l.loops)
                f = ∏(_loop_index(i, r), f)
            end
            push!(fs, f)
        end
    end
    return mul(fs...)
end

_iscontinuous_atom(a::IndexedRV) = iscontinuous(a.base)

"""
    plates(gm) -> Dict{RV,Vector{Index}}

The plates (loop indices) each variable of a model sits in: a `~` variable is
in the plates of the loops around its `~` lines, and an input is in a plate
when a `~` line indexes it with that plate's index (`sigma[j]`). Variables in
no plate map to an empty vector.
"""
function plates(gm::GenerativeModel)
    rvmap = Dict(v.name => v for v in gm.dag.nodes)
    out = Dict{RV,Vector{Index}}(v => Index[] for v in gm.dag.nodes)
    add!(v, I) = (I in out[v] || push!(out[v], I))
    lp = _lines_parents(gm)
    for (k, l) in enumerate(gm.lines)
        (l.kind === :tilde && haskey(rvmap, l.node)) || continue
        for (i, r) in l.loops
            add!(rvmap[l.node], _loop_index(i, r))
        end
        for a in _line_parents(l, lp[k], rvmap)
            a isa IndexedRV || continue
            I = a.index isa IndexShift ? a.index.index : a.index
            I isa Index && haskey(out, a.base) && add!(a.base, I)
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# Re-indexing and peeling a product
# ---------------------------------------------------------------------------

# A bound one less/more: integers and indices shift; symbols become `T - 1`.
_shiftbound(x::Integer, k) = x + k
_shiftbound(x::_IndexLike, k) = x + k
_shiftbound(s::Symbol, k) = k == 0 ? s : Expr(:call, k < 0 ? :- : :+, s, abs(k))
function _shiftbound(x::Expr, k)
    if x.head === :call && x.args[1] in (:+, :-) && length(x.args) == 3 && x.args[3] isa Integer
        total = (x.args[1] === :+ ? x.args[3] : -x.args[3]) + k
        return _shiftbound(x.args[2], total)
    end
    return Expr(:call, k < 0 ? :- : :+, x, abs(k))
end

# Replace the index `i` by `new` (an index, a shifted index or a number) in an
# index expression.
_reidx(x, i::Index, new) = x
_reidx(j::Index, i::Index, new) = j == i ? new : j
_reidx(s::IndexShift, i::Index, new) = s.index == i ? _shiftbound(new, s.k) : s
_reidx(r::IndexRange, i::Index, new) = IndexRange(_reidx(r.lo, i, new), _reidx(r.hi, i, new))

_reidx_atom(a::Atom, i, new) = Atom[a]
function _reidx_atom(a::IndexedRV, i, new)
    ix = _reidx(a.index, i, new)
    ix isa Integer && return Atom[a.base[ix]]
    if ix isa IndexRange && ix.lo isa Integer && ix.hi isa Integer
        return Atom[a.base[k] for k in ix.lo:ix.hi]
    end
    return Atom[IndexedRV(a.base, ix)]
end
_reidx_atom(c::Complement, i, new) = Atom[_complement(only(_reidx_atom(c.event, i, new)))]

_reindex(e::PExpr, i::Index, new) = mapprobs(p -> _mkprob(
    reduce(vcat, (_reidx_atom(a, i, new) for a in p.targets); init=Atom[]),
    reduce(vcat, (_reidx_atom(a, i, new) for a in p.given); init=Atom[]); name=p.name), e)

"""
    peel(∏_{s=lo}^{hi} f(s)) -> [∏_{s=lo}^{hi-1} f(s)] f(hi)

Split the last factor off a product over a plate (`hi` may be a number, an
index such as `t`, or a symbol such as `T`, which becomes `T - 1`). With a
symbolic `hi` the remaining product keeps the same index name.
"""
function peel(p::PlateProduct)
    i = p.index
    hi1 = _shiftbound(i.hi, -1)
    j = Index(i.name, i.lo, hi1)
    rest = i.lo isa Integer && hi1 isa Integer && hi1 < i.lo ? Literal(1) : ∏(j, _reindex(p.body, i, j))
    return mul(rest, _reindex(p.body, i, i.hi isa Union{Integer,_IndexLike} ? i.hi : _boundindex(i.hi)))
end
# A symbolic end (`T`) used as an index value: an Index named after it.
_boundindex(s::Symbol) = Index(s, s, s)
_boundindex(x) = throw(ArgumentError("cannot peel a product whose end is $(x)"))

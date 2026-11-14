# Reasoning from assumed facts (types in facts.jl):
#
# * simplify rules that use independence, disjointness and implication facts;
# * `deduce`, which finds the value of a probability from known values and
#   facts, by linear algebra over the 2ⁿ cells of n events (independence
#   facts add product equations once enough is known).

# ---------------------------------------------------------------------------
# Facts as predicates
# ---------------------------------------------------------------------------

_isdisjoint(f::Facts, a::Atom, b::Atom) =
    a != b && any(d -> a in d.parts && b in d.parts, f.disjoint)

# a ⊆ b, directly or by contraposition (bᶜ ⊆ aᶜ)
_implies(f::Facts, a::Atom, b::Atom) = a == b ||
    any(i -> (i.a == a && i.b == b) ||
             (iseventlike(a) && iseventlike(b) && i.a == complement(b) && i.b == complement(a)), f.implies)

# Does an assumed independence give T ⟂ W | rest? Decomposition and weak union:
# X ⟂ Y | Z gives X₁ ⟂ Y₁ | Z ∪ Y₂ for X₁ ⊆ X and Y = Y₁ ∪ Y₂ (and symmetrically).
function _assumed_indep(f::Facts, T::Vector{RV}, W::Vector{RV}, rest::Vector{RV})
    (isempty(T) || isempty(W)) && return nothing
    for s in f.independent
        for (X, Y) in ((s.X, s.Y), (s.Y, s.X))
            issubset(T, X) && issubset(W, Y) || continue
            issubset(s.Z, rest) || continue
            issubset(rest, union(s.Z, setdiff(Y, W), setdiff(X, T))) || continue
            return s
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Simplification rules
# ---------------------------------------------------------------------------

function rule_fact_disjoint(p::Prob, ctx::SimplifyContext)
    f = ctx.facts
    isempty(f.disjoint) && return nothing
    for t in p.targets, x in vcat(p.targets, p.given)
        if _isdisjoint(f, t, x)
            return (Literal(0), "$(_astr(t)) and $(_astr(x)) are disjoint (assumed)")
        end
    end
    # P(X ∩ (A ∪ B) | C) = P(X ∩ A | C) + P(X ∩ B | C) for disjoint A, B
    for (k, u) in enumerate(p.targets)
        u isa EventUnion || continue
        all(i -> all(j -> i == j || _isdisjoint(f, u.parts[i], u.parts[j]), eachindex(u.parts)), eachindex(u.parts)) || continue
        others = Atom[t for (m, t) in enumerate(p.targets) if m != k]
        return (add((_mkprob(vcat(others, x), p.given; name=p.name) for x in u.parts)...),
                "addition rule for disjoint events")
    end
    return nothing
end
rule_fact_disjoint(::PExpr, ::SimplifyContext) = nothing

function rule_fact_implies(p::Prob, ctx::SimplifyContext)
    f = ctx.facts
    isempty(f.implies) && return nothing
    atoms = vcat(p.targets, p.given)
    for b in p.targets, a in atoms
        a == b && continue
        if _implies(f, a, b)
            keep = Atom[t for t in p.targets if t != b]
            new = isempty(keep) ? Literal(1) : _like(p, keep, p.given)
            return (new, "$(_astr(a)) implies $(_astr(b)) (assumed)")
        end
    end
    # A ∪ B = B when A ⊆ B
    for (k, u) in enumerate(p.targets)
        u isa EventUnion || continue
        for x in u.parts, y in u.parts
            x == y && continue
            _implies(f, x, y) || continue
            newu = _evunion(Atom[z for z in u.parts if z != x])
            ts = Atom[m == k ? newu : t for (m, t) in enumerate(p.targets)]
            return (_mkprob(ts, p.given; name=p.name), "$(_astr(x)) implies $(_astr(y)) (assumed), so $(_astr(u)) = $(_astr(newu))")
        end
    end
    return nothing
end
rule_fact_implies(::PExpr, ::SimplifyContext) = nothing

function rule_fact_independence(p::Prob, ctx::SimplifyContext)
    f = ctx.facts
    isempty(f.independent) && return nothing
    any(a -> a isa Intervention, p.given) && return nothing
    T = _rvsof(p.targets)
    # drop a conditioning atom
    for w in p.given
        rest = _rvsof(a for a in p.given if a != w)
        s = _assumed_indep(f, T, rvs(w), rest)
        s === nothing && continue
        return (_like(p, p.targets, Atom[a for a in p.given if a != w]), "$(sprint(show, s)) (assumed)")
    end
    # split a joint: P(T₁ ∩ T₂ | G) = P(T₁ | G) P(T₂ | G)
    length(p.targets) >= 2 || return nothing
    G = _rvsof(p.given)
    for k in 1:length(p.targets)-1, T1 in _combinations(p.targets, k)
        T2 = Atom[t for t in p.targets if !(t in T1)]
        s = _assumed_indep(f, _rvsof(T1), _rvsof(T2), G)
        s === nothing && continue
        return (mul(_like(p, T1, p.given), _like(p, T2, p.given)),
                "$(sprint(show, s)) (assumed): the joint factorises")
    end
    return nothing
end
rule_fact_independence(::PExpr, ::SimplifyContext) = nothing

push!(_RULES, rule_fact_disjoint, rule_fact_implies, rule_fact_independence)

# ---------------------------------------------------------------------------
# deduce
# ---------------------------------------------------------------------------

"""
    Solution

The result of [`deduce`](@ref): the query, its value (or `NaN`), and whether
the facts determine it. Converts to `Float64`.
"""
struct Solution
    query::PExpr
    value::Float64
    determined::Bool
    undetermined::Vector{PExpr}
end
Base.Float64(s::Solution) = s.value
function Base.show(io::IO, s::Solution)
    show(io, s.query)
    if s.determined
        print(io, " = ", round(s.value; sigdigits=10))
    else
        print(io, " is not determined by the facts")
        isempty(s.undetermined) || print(io, " (unknown: ", join(map(x -> sprint(show, x), s.undetermined), ", "), ")")
    end
end
Base.show(io::IO, ::MIME"text/latex", s::Solution) =
    print(io, "\$\$", _latex(s.query), s.determined ? " = $(round(s.value; sigdigits=10))" : " \\text{ (not determined)}", "\$\$")

# The linear system over the cells of the event space.
mutable struct _EventSpace
    vars::Vector{RV}
    A::Matrix{Float64}
    b::Vector{Float64}
end

_ncells(S::_EventSpace) = 2^length(S.vars)

# Cell k (1-based) has events[i] happening iff bit i-1 of k-1 is set.
_cellvals(vars, k) = Dict{RV,Any}(v => isodd((k - 1) >> (i - 1)) for (i, v) in enumerate(vars))

function _cellvec(S::_EventSpace, atoms)
    c = zeros(_ncells(S))
    for k in 1:_ncells(S)
        vals = _cellvals(S.vars, k)
        c[k] = all(a -> _holds(a, vals), atoms) ? 1.0 : 0.0
    end
    return c
end

function _addrow!(S::_EventSpace, c, rhs)
    S.A = vcat(S.A, reshape(c, 1, :))
    push!(S.b, rhs)
end

# The probability term(s) of a known value, as a linear form c⋅p = rhs.
function _linform(S::_EventSpace, e::PExpr)
    e isa Literal && return (fill(Float64(e.value), _ncells(S)))     # c ⋅ p = c, since ∑p = 1
    if e isa Prob
        isempty(e.given) || throw(ArgumentError("only unconditional terms inside a sum: $(e)"))
        return _cellvec(S, e.targets)
    end
    if e isa Sum
        return reduce(+, (_linform(S, t) for t in e.terms))
    end
    if e isa Product
        c, r = _coef(e)
        r isa Prob && return c .* _linform(S, r)
    end
    throw(ArgumentError("cannot use $(e) as a linear fact; give probabilities, sums of them, or one conditional probability"))
end

function _addvalue!(S::_EventSpace, e::PExpr, v::Float64)
    if e isa Prob && !isempty(e.given)          # P(T | G) = v  ⇔  P(T ∩ G) - v P(G) = 0
        _addrow!(S, _cellvec(S, vcat(e.targets, e.given)) .- v .* _cellvec(S, e.given), 0.0)
    else
        _addrow!(S, _linform(S, e), v)
    end
end

function _eventvars(xs)
    vs = RV[]
    for x in xs
        for v in x
            isevent(v) || throw(ArgumentError(
                "deduce works with named events (@event A B); $(v.name) is a $(v.kind) variable"))
            v in vs || push!(vs, v)
        end
    end
    return vs
end

_exprvars(e::PExpr) = allvars(e)

# Current solution state: a particular solution and the null space.
function _state(S::_EventSpace)
    p0 = pinv(S.A) * S.b
    res = maximum(abs, S.A * p0 .- S.b; init=0.0)
    res > 1e-8 && throw(ArgumentError("the facts are inconsistent (no probabilities satisfy them all)"))
    N = nullspace(S.A)
    return p0, N
end
_determined(N, c) = maximum(abs, N' * c; init=0.0) < 1e-9

# Independence facts as product equations p(x,y,z) p(z) = p(x,z) p(y,z),
# added as linear rows whenever three of the four are already determined.
function _propagate_independence!(S::_EventSpace, f::Facts)
    for _ in 1:50
        p0, N = _state(S)
        added = false
        for s in f.independent
            for xa in _assignments_of(s.X), ya in _assignments_of(s.Y), za in _assignments_of(s.Z)
                cxyz = _cellvec(S, vcat(xa, ya, za))
                cz = _cellvec(S, za)
                cxz = _cellvec(S, vcat(xa, za))
                cyz = _cellvec(S, vcat(ya, za))
                d = map(c -> _determined(N, c), (cxyz, cz, cxz, cyz))
                count(d) == 3 || continue
                val(c) = dot(c, p0)
                if !d[1]
                    pz = val(cz)
                    _addrow!(S, cxyz, pz > 1e-12 ? val(cxz) * val(cyz) / pz : 0.0)
                elseif !d[2] && val(cxyz) > 1e-12
                    _addrow!(S, cz, val(cxz) * val(cyz) / val(cxyz))
                elseif !d[3] && val(cyz) > 1e-12
                    _addrow!(S, cxz, val(cxyz) * val(cz) / val(cyz))
                elseif !d[4] && val(cxz) > 1e-12
                    _addrow!(S, cyz, val(cxyz) * val(cz) / val(cxz))
                else
                    continue
                end
                added = true
            end
        end
        added || return S
    end
    return S
end

# Every truth assignment of a set of event variables, as lists of atoms.
function _assignments_of(vs::Vector{RV})
    out = [Atom[]]
    for v in vs
        out = vcat([vcat(a, Atom[v]) for a in out], [vcat(a, Atom[complement(v)]) for a in out])
    end
    return out
end

# The event variables the facts (and any extra expressions) mention.
function _factvars(f::Facts, extra...)
    exprs = [first(kv) for kv in f.values]
    return _eventvars(vcat([allvars(e) for e in exprs],
                           [vcat(s.X, s.Y, s.Z) for s in f.independent],
                           [unique(RV[v for a in d.parts for v in rvs(a)]) for d in f.disjoint],
                           [vcat(rvs(i.a), rvs(i.b)) for i in f.implies],
                           [allvars(e) for e in extra]))
end

_eventspace(f::Facts, extra...) = _eventspace_on(_factvars(f, extra...), f)

function _eventspace_on(vars::Vector{RV}, f::Facts)
    length(vars) > 12 && throw(ArgumentError("deduce handles up to 12 events, got $(length(vars))"))
    S = _EventSpace(vars, ones(1, 2^length(vars)), [1.0])
    for (e, v) in f.values
        _addvalue!(S, e, v)
    end
    for d in f.disjoint, i in eachindex(d.parts), j in i+1:length(d.parts)
        _addrow!(S, _cellvec(S, Atom[d.parts[i], d.parts[j]]), 0.0)
    end
    for i in f.implies
        _addrow!(S, _cellvec(S, Atom[i.a, complement(i.b)]), 0.0)
    end
    _propagate_independence!(S, f)
    return S
end

function _solve_value(e::PExpr, S, p0, N, missingterms)
    if e isa Literal
        return Float64(e.value)
    elseif e isa Prob
        num = _cellvec(S, vcat(e.targets, e.given))
        ok = _determined(N, num)
        if isempty(e.given)
            ok || (push!(missingterms, e); return NaN)
            return dot(num, p0)
        end
        den = _cellvec(S, e.given)
        okd = _determined(N, den)
        (ok && okd) || (push!(missingterms, ok ? Prob(e.given, Atom[], e.name) : e); return NaN)
        return dot(num, p0) / dot(den, p0)
    elseif e isa Sum
        return sum(_solve_value(t, S, p0, N, missingterms) for t in e.terms)
    elseif e isa Product
        return prod(_solve_value(t, S, p0, N, missingterms) for t in e.factors)
    elseif e isa Quotient
        return _solve_value(e.num, S, p0, N, missingterms) / _solve_value(e.den, S, p0, N, missingterms)
    end
    throw(ArgumentError("deduce cannot evaluate $(e)"))
end

"""
    deduce(query, facts...) -> Solution

The value of a probability expression implied by known values and facts about
named events. Facts are as for [`assume`](@ref):

```julia
@event A B
deduce(P(A ∩ B), P(A) => 0.3, P(B) => 0.5, P(A ∪ B) => 0.6)   # 0.2
deduce(P(A ∪ B), P(A) => 0.3, P(B) => 0.4, A ⟂ B)             # 0.58
deduce(P(A | B), P(B | A) => 0.9, P(A) => 0.01, P(B | A') => 0.05)
```

The known values are linear constraints on the probabilities of the 2ⁿ cells
of the events; [`disjoint`](@ref) and [`implies`](@ref) add more, and
independence adds product equations as soon as enough is known. If the facts
do not pin the query down, the result says so (`s.determined == false`) and
names the unknown terms. Inconsistent facts are an error.
"""
function deduce(query::PExpr, facts...)
    f = assume(facts...)
    S = _eventspace(f, query)
    p0, N = _state(S)
    missingterms = PExpr[]
    v = _solve_value(query, S, p0, N, missingterms)
    ok = isempty(missingterms)
    return Solution(query, ok ? v : NaN, ok, unique(missingterms))
end

"""
    cell_probabilities(facts...; events = nothing) -> (events, probs)

The probability of every cell of the Venn diagram of the events, as far as the
facts determine it. Returns the events in order, and a vector of length 2ⁿ
whose entry `k` is the probability of the cell where `events[i]` happens iff
bit `i - 1` of `k - 1` is set, or `nothing` where the facts leave it open.
`events` fixes the order (and may add events the facts don't mention).
"""
function cell_probabilities(facts...; events=nothing)
    f = assume(facts...)
    vars = _factvars(f)
    if events !== nothing
        ev = _eventvars([RV[v for v in events]])
        all(v -> v in ev, vars) || throw(ArgumentError("`events` must include every event in the facts"))
        vars = ev
    end
    S = _eventspace_on(vars, f)
    p0, N = _state(S)
    probs = Union{Nothing,Float64}[]
    for k in 1:_ncells(S)
        c = zeros(_ncells(S)); c[k] = 1
        push!(probs, _determined(N, c) ? max(0.0, dot(c, p0)) : nothing)
    end
    return (S.vars, probs)
end

function Base.show(io::IO, ::MIME"text/plain", f::Facts)
    print(io, "Facts (", _nfacts(f), ")")
    for i in f.iid
        print(io, "\n  ", i.base.name, "[i] iid", isempty(i.given) ? "" : " given " * _atomlist(i.given))
    end
    for m in f.markov
        print(io, "\n  ", m.base.name, " is Markov")
    end
    for s in f.independent
        print(io, "\n  ", sprint(show, s))
    end
    for d in f.disjoint
        print(io, "\n  disjoint(", join(map(_astr, d.parts), ", "), ")")
    end
    for i in f.implies
        print(io, "\n  ", _astr(i.a), " ⊆ ", _astr(i.b))
    end
    for (e, v) in f.values
        print(io, "\n  ", sprint(show, e), " = ", v)
    end
end
Base.show(io::IO, f::Facts) = print(io, "Facts(", _nfacts(f), ")")

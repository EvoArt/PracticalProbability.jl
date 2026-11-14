# Expectations, variances and covariances.
#
# Random quantities (`X + 2Y`, `X^2`, `𝟙(A)`, `exp(X)`, or another expectation
# such as `E[Y | X]`, which is a function of X) form their own small algebra,
# `RExpr`. `E[f | Z]`, `Var(f | Z)` and `Cov(f, g | Z)` are probability
# expressions (`PExpr`) like `P(⋯)`, so they combine with probabilities and
# go through `simplify`, `latex` and `compute`.

# ---------------------------------------------------------------------------
# Random quantities
# ---------------------------------------------------------------------------

"""
    RExpr

A random quantity: a function of random variables, built with arithmetic on
[`RV`](@ref)s (`X + 2Y`, `X * Y`, `X^2`, `exp(X)`), indicators [`𝟙`](@ref)`(A)`,
and expectations treated as random (`E(Y | X)` is a function of `X`).
"""
abstract type RExpr end
struct RVal <: RExpr
    rv::RV
end
struct RConst <: RExpr
    value::Number
end
struct RAdd <: RExpr
    terms::Vector{RExpr}
end
struct RMul <: RExpr
    factors::Vector{RExpr}
end
struct RPow <: RExpr
    base::RExpr
    p::Number
end
struct RFun <: RExpr
    f::Symbol
    arg::RExpr
end
struct RInd <: RExpr
    event::Atom
end
struct RQ <: RExpr
    e::PExpr
end

Base.:(==)(a::RVal, b::RVal) = a.rv == b.rv
Base.:(==)(a::RConst, b::RConst) = a.value == b.value
Base.:(==)(a::RAdd, b::RAdd) = _multiset_eq(a.terms, b.terms)
Base.:(==)(a::RMul, b::RMul) = _multiset_eq(a.factors, b.factors)
Base.:(==)(a::RPow, b::RPow) = a.base == b.base && a.p == b.p
Base.:(==)(a::RFun, b::RFun) = a.f === b.f && a.arg == b.arg
Base.:(==)(a::RInd, b::RInd) = a.event == b.event
Base.:(==)(a::RQ, b::RQ) = a.e == b.e
Base.:(==)(::RExpr, ::RExpr) = false
Base.hash(a::RVal, h::UInt) = hash(a.rv, hash(:RVal, h))
Base.hash(a::RConst, h::UInt) = hash(a.value, hash(:RConst, h))
Base.hash(a::RAdd, h::UInt) = hash(_multiset_hash(a.terms), hash(:RAdd, h))
Base.hash(a::RMul, h::UInt) = hash(_multiset_hash(a.factors), hash(:RMul, h))
Base.hash(a::RPow, h::UInt) = hash(a.p, hash(a.base, hash(:RPow, h)))
Base.hash(a::RFun, h::UInt) = hash(a.arg, hash(a.f, hash(:RFun, h)))
Base.hash(a::RInd, h::UInt) = hash(a.event, hash(:RInd, h))
Base.hash(a::RQ, h::UInt) = hash(a.e, hash(:RQ, h))

_r(x::RV) = (isevent(x) ? RInd(x) : RVal(x))
_r(x::RExpr) = x
_r(x::Number) = RConst(x)
_r(e::PExpr) = e isa Literal ? RConst(e.value) : RQ(e)

# coefficient × rest
function _rcoef(t::RExpr)
    t isa RConst && return (t.value, RConst(1))
    if t isa RMul && t.factors[1] isa RConst
        rest = t.factors[2:end]
        return (t.factors[1].value, length(rest) == 1 ? rest[1] : RMul(rest))
    end
    return (1, t)
end

function radd(xs::RExpr...)
    rests = RExpr[]
    coefs = Number[]
    for x in xs, t in (x isa RAdd ? x.terms : (x,))
        c, r = _rcoef(t)
        i = findfirst(==(r), rests)
        i === nothing ? (push!(rests, r); push!(coefs, c)) : (coefs[i] += c)
    end
    terms = RExpr[]
    for (c, r) in zip(coefs, rests)
        c == 0 && continue
        push!(terms, r == RConst(1) ? RConst(c) : rmul(RConst(c), r))
    end
    isempty(terms) && return RConst(0)
    length(terms) == 1 && return terms[1]
    return RAdd(terms)
end

# base and exponent of a factor
_rbase(x::RPow) = (x.base, x.p)
_rbase(x::RExpr) = (x, 1)

function rmul(xs::RExpr...)
    c = 1
    bases = RExpr[]
    exps = Number[]
    for x in xs, f in (x isa RMul ? x.factors : (x,))
        if f isa RConst
            c *= f.value
        else
            b, p = _rbase(f)
            i = findfirst(==(b), bases)
            i === nothing ? (push!(bases, b); push!(exps, p)) : (exps[i] += p)
        end
    end
    c == 0 && return RConst(0)
    c isa Complex && iszero(imag(c)) && (c = real(c))
    fs = RExpr[rpow(b, p) for (b, p) in zip(bases, exps) if p != 0]

    c != 1 && pushfirst!(fs, RConst(c))
    isempty(fs) && return RConst(c)
    length(fs) == 1 && return fs[1]
    return RMul(fs)
end

function rpow(b::RExpr, p::Number)
    p isa Rational && isone(denominator(p)) && (p = numerator(p))     # 2//1 is the integer 2
    p == 0 && return RConst(1)

    p == 1 && return b
    b isa RConst && return RConst(_numpow(b.value, p))
    b isa RPow && return rpow(b.base, b.p * p)
    b isa RInd && p > 0 && return b                # 𝟙(A)ᵖ = 𝟙(A)
    b isa RFun && b.f === :exp && return RFun(:exp, rmul(RConst(p), b.arg))   # (eᵃ)ᵖ = e^{pa}

    return RPow(b, p)
end

# Powers of numbers that stay exact (2^-1 is 1//2, not a DomainError).
_numpow(a::Union{Integer,Rational}, p::Integer) = p < 0 ? (a // 1)^p : a^p
_numpow(a::Union{Integer,Rational}, p::Rational) = isinteger(p) ? _numpow(a, Integer(p)) : float(a)^p
_numpow(a, p) = a^p

const _R = Union{RV,RExpr}

Base.:+(a::_R, b::_R) = radd(_r(a), _r(b))
Base.:+(a::Number, b::_R) = radd(_r(a), _r(b))
Base.:+(a::_R, b::Number) = radd(_r(a), _r(b))
Base.:-(a::_R) = rmul(RConst(-1), _r(a))
Base.:-(a::_R, b::_R) = radd(_r(a), -_r(b))
Base.:-(a::Number, b::_R) = radd(_r(a), -_r(b))
Base.:-(a::_R, b::Number) = radd(_r(a), RConst(-b))
Base.:*(a::_R, b::_R) = rmul(_r(a), _r(b))
Base.:*(a::Number, b::_R) = rmul(_r(a), _r(b))
Base.:*(a::_R, b::Number) = rmul(_r(a), _r(b))
Base.:/(a::_R, b::Number) = rmul(_r(a), RConst(_div(1, b)))
Base.:/(a::_R, b::_R) = rmul(_r(a), rpow(_r(b), -1))
Base.:/(a::Number, b::_R) = rmul(RConst(a), rpow(_r(b), -1))
Base.:^(a::_R, p::Number) = rpow(_r(a), p)
Base.:^(a::_R, p::Integer) = rpow(_r(a), p)
for f in (:exp, :log, :sqrt, :abs)
    @eval Base.$f(x::_R) = RFun($(QuoteNode(f)), _r(x))
end

"""
    𝟙(A)
    indicator(A)

The indicator of an event: 1 when `A` happens, 0 otherwise. `E[𝟙(A)] = P(A)`.
(`𝟙` is typed `\\bbone<tab>`.)
"""
𝟙(a::Atom) = RInd(_check_event(a))
const indicator = 𝟙

# The variables a random quantity depends on.
_rvars(x::RVal) = Set{RV}([x.rv])
_rvars(::RConst) = Set{RV}()
_rvars(x::RAdd) = mapreduce(_rvars, union, x.terms; init=Set{RV}())
_rvars(x::RMul) = mapreduce(_rvars, union, x.factors; init=Set{RV}())
_rvars(x::RPow) = _rvars(x.base)
_rvars(x::RFun) = _rvars(x.arg)
_rvars(x::RInd) = Set{RV}(rvs(x.event))
_rvars(x::RQ) = _freevars(x.e)

# `f | Z` for a random quantity
struct GivenR
    f::RExpr
    given::Vector{Atom}
end
Base.:|(f::RExpr, g::Atom) = GivenR(f, Atom[g])
Base.:|(f::RExpr, g::Tuple) = GivenR(f, _atoms(g))

# ---------------------------------------------------------------------------
# Expectation, variance, covariance as probability expressions
# ---------------------------------------------------------------------------

"""
    Expect(f, given, name)

`E[f | given]` under the distribution `name`. Build with [`E`](@ref).
"""
struct Expect <: PExpr
    f::RExpr
    given::Vector{Atom}
    name::Symbol
end
"""
    Variance(f, given, name)

`Var(f | given)`. Build with [`Var`](@ref).
"""
struct Variance <: PExpr
    f::RExpr
    given::Vector{Atom}
    name::Symbol
end
"""
    Covariance(f, g, given, name)

`Cov(f, g | given)`. Build with [`Cov`](@ref).
"""
struct Covariance <: PExpr
    f::RExpr
    g::RExpr
    given::Vector{Atom}
    name::Symbol
end
"""
    Value(f)

The value of a random quantity, inside an expression: a function of its
variables (as in `∑_X X P(X)`).
"""
struct Value <: PExpr
    f::RExpr
end
"""
    Power(base, p)

`base^p` for a probability expression, e.g. `E[X]²`. Build with `^`.
"""
struct Power <: PExpr
    base::PExpr
    p::Number
end

Base.:^(e::PExpr, p::Integer) = _pexpr_pow(e, p)
Base.:^(e::PExpr, p::Number) = _pexpr_pow(e, p)
function _pexpr_pow(e::PExpr, p)
    p == 1 && return e
    p == 0 && return Literal(1)
    e isa Literal && return Literal(e.value^p)
    e isa Power && return Power(e.base, e.p * p)
    return Power(e, p)
end

# The quantity and conditioning from the arguments of E(⋯), Var(⋯).
function _quantity(args)
    isempty(args) && throw(ArgumentError("E/Var need a random quantity"))
    first_, rest = args[1], args[2:end]
    given = Atom[]
    if first_ isa GivenR
        f = first_.f
        append!(given, first_.given)
    elseif first_ isa Given
        f = _targets_quantity(first_.targets)
        append!(given, first_.given)
    else
        f = _r(first_)
    end
    for a in rest
        append!(given, a isa Atom ? Atom[a] : _atoms(a))
    end
    return f, _flatten_inter(given)
end
function _targets_quantity(ts)
    length(ts) == 1 && ts[1] isa RV && !isevent(ts[1]) && return RVal(ts[1])
    all(iseventlike, ts) && return RInd(_evintersect(ts))
    throw(ArgumentError("E(⋯ | ⋯) takes one random quantity before the bar; got $(join(map(_astr, ts), ", "))"))
end

"""
    E(f)
    E(f | Z)
    E(X^2 + 2Y | Z, W; name = :P)
    𝔼(…)

The expectation of a random quantity, optionally conditional. `f` is a
variable, an arithmetic combination of variables (`X + 2Y`, `X * Y`, `X^2`,
`exp(X)`), an indicator [`𝟙`](@ref)`(A)` (an event on its own means its
indicator), or another expectation (`E(E(Y | X))`). `name` names the
distribution (`E(f; name = :q)` is written `E_q[f]`).

Rules (in [`simplify`](@ref)): linearity, `E[c] = c`, `E[𝟙(A)] = P(A)`,
taking out what is known (`E[g(Z) Y | Z] = g(Z) E[Y | Z]`), the tower property
(`E[E[Y | X, Z] | Z] = E[Y | Z]`), and `E[X Y] = E[X] E[Y]` for independent
`X`, `Y` (by a graph or assumed facts). Explicit: [`definition`](@ref),
[`tower`](@ref). Numbers: [`compute`](@ref) on a network whose values are
numbers.
"""
function E(args...; name::Symbol=:P)
    f, given = _quantity(args)
    return Expect(f, given, name)
end
const 𝔼 = E

"""
    Var(f | Z; name = :P)

The (conditional) variance of a random quantity. See also [`definition`](@ref)
(`E[f²] - E[f]²`) and [`total_variance`](@ref).
"""
function Var(args...; name::Symbol=:P)
    f, given = _quantity(args)
    return Variance(f, given, name)
end

"""
    Cov(f, g | Z; name = :P)

The (conditional) covariance of two random quantities.
"""
function Cov(f, g, rest...; name::Symbol=:P)
    gq, given = _quantity((g, rest...))
    fq = f isa GivenR ? f.f : _r(f)
    return Covariance(fq, gq, given, name)
end

for T in (:Expect, :Variance)
    @eval Base.:(==)(a::$T, b::$T) = a.f == b.f && issetequal(a.given, b.given) && a.name === b.name
    @eval Base.hash(a::$T, h::UInt) = hash(a.name, hash(Set(a.given), hash(a.f, hash($(QuoteNode(T)), h))))
end
Base.:(==)(a::Covariance, b::Covariance) = a.name === b.name && issetequal(a.given, b.given) &&
    ((a.f == b.f && a.g == b.g) || (a.f == b.g && a.g == b.f))
Base.hash(a::Covariance, h::UInt) = hash(a.name, hash(Set(a.given), hash(hash(a.f) + hash(a.g), hash(:Covariance, h))))
Base.:(==)(a::Value, b::Value) = a.f == b.f
Base.hash(a::Value, h::UInt) = hash(a.f, hash(:Value, h))
Base.:(==)(a::Power, b::Power) = a.base == b.base && a.p == b.p
Base.hash(a::Power, h::UInt) = hash(a.p, hash(a.base, hash(:Power, h)))

const _Moment = Union{Expect,Variance,Covariance}

# An expectation is itself a random quantity (a function of what it conditions
# on), so it can be conditioned and averaged again: `E(E(Y | X, Z) | Z)`.
Base.:|(e::_Moment, g::Atom) = GivenR(RQ(e), Atom[g])
Base.:|(e::_Moment, g::Tuple) = GivenR(RQ(e), _atoms(g))
_givenfree(e::_Moment) = Set{RV}(v for v in (freerv(a isa Intervention ? a.target : a) for a in e.given) if v !== nothing)
_freevars(e::_Moment) = _givenfree(e)
_allvars(e::Expect) = union(_rvars(e.f), Set(_rvsof(e.given)))
_allvars(e::Variance) = union(_rvars(e.f), Set(_rvsof(e.given)))
_allvars(e::Covariance) = union(_rvars(e.f), _rvars(e.g), Set(_rvsof(e.given)))
_freevars(v::Value) = _rvars(v.f)
_allvars(v::Value) = _rvars(v.f)
_freevars(p::Power) = _freevars(p.base)
_allvars(p::Power) = _allvars(p.base)

children(::Union{_Moment,Value}) = PExpr[]
rebuild(e::Union{_Moment,Value}, cs) = e
children(p::Power) = PExpr[p.base]
rebuild(p::Power, cs) = cs[1]^p.p

_subs(e::Expect, d) = Expect(e.f, Atom[_sa(a, d) for a in e.given], e.name)
_subs(e::Variance, d) = Variance(e.f, Atom[_sa(a, d) for a in e.given], e.name)
_subs(e::Covariance, d) = Covariance(e.f, e.g, Atom[_sa(a, d) for a in e.given], e.name)
_subs(v::Value, d) = v
_subs(p::Power, d) = _subs(p.base, d)^p.p

_atomic(::Union{_Moment,Value}) = true
_atomic(p::Power) = _atomic(p.base)

# ---------------------------------------------------------------------------
# Explicit transformations
# ---------------------------------------------------------------------------

"""
    definition(E[f | Z]) -> ∑_x f(x) P(x | Z)       (∫ … p(x | Z) dx if continuous)
    definition(Var(f | Z)) -> E[f² | Z] - E[f | Z]²
    definition(Cov(f, g | Z)) -> E[f g | Z] - E[f | Z] E[g | Z]

Write an expectation, variance or covariance out by its definition.
"""
function definition(e::Expect)
    vs = sort!(collect(setdiff(_rvars(e.f), _givenfree(e))))
    isempty(vs) && return Value(e.f)
    return marg(vs, mul(Value(e.f), Prob(Atom[vs...], e.given, e.name)))
end
definition(v::Variance) = add(Expect(rpow(v.f, 2), v.given, v.name), -(Expect(v.f, v.given, v.name)^2))
definition(c::Covariance) = add(Expect(rmul(c.f, c.g), c.given, c.name),
                                -mul(Expect(c.f, c.given, c.name), Expect(c.g, c.given, c.name)))

"""
    tower(E[f | Z], Y) -> E[ E[f | Y, Z] | Z ]

The law of total expectation (tower property), conditioning on `Y` inside.
"""
tower(e::Expect, Y) = Expect(RQ(Expect(e.f, vcat(_aslist(Y), e.given), e.name)), e.given, e.name)

"""
    total_variance(Var(f | Z), Y) -> E[Var(f | Y, Z) | Z] + Var(E[f | Y, Z] | Z)

The law of total variance (Eve's law).
"""
function total_variance(v::Variance, Y)
    inner = vcat(_aslist(Y), v.given)
    return add(Expect(RQ(Variance(v.f, inner, v.name)), v.given, v.name),
               Variance(RQ(Expect(v.f, inner, v.name)), v.given, v.name))
end

# ---------------------------------------------------------------------------
# Simplification rules
# ---------------------------------------------------------------------------

# Are the variables A and B independent given C under distribution `name`,
# by its graph or by assumed facts?
function _independent(ctx::SimplifyContext, name::Symbol, A::Vector{RV}, B::Vector{RV}, C::Vector{RV})
    (isempty(A) || isempty(B)) && return nothing
    g = get(ctx.dags, name, nothing)
    if g !== nothing && all(v -> v in g.nodes, vcat(A, B, C)) && isempty(intersect(A, B))
        d_separated(g, A, B, C) && return "$(_str(A)) ⟂ $(_str(B))$(_givenstr(C)) (d-separation)"
    end
    s = _assumed_indep(ctx.facts, A, B, C)
    s === nothing || return "$(sprint(show, s)) (assumed)"
    _laws_independent(ctx.laws, name, A, B) &&                   # transforms.jl
        return string(_str(A), " ⟂ ", _str(B), " (independent laws)")
    return nothing

end

_rfactors(f::RExpr) = f isa RMul ? f.factors : RExpr[f]

function rule_expect(e::Expect, ctx::SimplifyContext)
    f = e.f
    f isa RConst && return (Literal(f.value), "the expectation of a constant")
    f isa RAdd && return (add((Expect(t, e.given, e.name) for t in f.terms)...), "linearity of expectation")
    c, r = _rcoef(f)
    c != 1 && return (mul(Literal(c), Expect(r, e.given, e.name)), "linearity of expectation")
    f isa RInd && return (_mkprob(Atom[f.event], e.given; name=e.name), "the expectation of an indicator is a probability")
    known = _givenfree(e)
    # E[g(Z) | Z] = g(Z);  E[g(Z) h | Z] = g(Z) E[h | Z]
    if issubset(_rvars(f), known) && !isempty(e.given)
        return (Value(f), "taking out what is known")
    end
    fs = _rfactors(f)
    if length(fs) > 1
        kn = RExpr[x for x in fs if issubset(_rvars(x), known)]
        if !isempty(kn)
            un = RExpr[x for x in fs if !issubset(_rvars(x), known)]
            return (mul(Value(rmul(kn...)), Expect(rmul(un...), e.given, e.name)), "taking out what is known")
        end
    end
    # tower: E[E[g | G₂] | G₁] = E[g | G₁] when G₁ ⊆ G₂
    if f isa RQ && f.e isa Expect && f.e.name === e.name && issubset(_atomset(e.given), _atomset(f.e.given))
        return (Expect(f.e.f, e.given, e.name), "tower property")
    end
    G = _rvsof(e.given)
    any(a -> a isa Intervention, e.given) && return nothing
    # independence: E[g h | Z] = E[g | Z] E[h | Z]
    if length(fs) > 1
        for k in 1:length(fs)-1, part in _combinations(fs, k)
            other = RExpr[x for x in fs if !(x in part)]       # rmul merges repeats, so factors are distinct
            A =sort!(collect(mapreduce(_rvars, union, part)))
            B = sort!(collect(mapreduce(_rvars, union, other)))
            isempty(intersect(A, B)) || continue
            why = _independent(ctx, e.name, A, B, G)
            why === nothing && continue
            return (mul(Expect(rmul(part...), e.given, e.name), Expect(rmul(other...), e.given, e.name)),
                    "independence: " * why)
        end
    end
    # drop a conditioning atom independent of f
    F = sort!(collect(_rvars(f)))
    for w in e.given
        rest = _rvsof(a for a in e.given if a != w)
        any(v -> v in F, rvs(w)) && continue
        why = _independent(ctx, e.name, F, rvs(w), rest)
        why === nothing && continue
        return (Expect(f, Atom[a for a in e.given if a != w], e.name), why)
    end
    return nothing
end
rule_expect(::PExpr, ::SimplifyContext) = nothing

function rule_variance(v::Variance, ctx::SimplifyContext)
    f = v.f
    f isa RConst && return (Literal(0), "the variance of a constant is 0")
    !isempty(v.given) && issubset(_rvars(f), _givenfree(v)) &&
        return (Literal(0), "a known quantity has variance 0")
    c, r = _rcoef(f)
    c != 1 && return (mul(Literal(c^2), Variance(r, v.given, v.name)), "Var(a X) = a² Var(X)")
    if f isa RAdd
        consts = RExpr[t for t in f.terms if t isa RConst]
        if !isempty(consts)
            return (Variance(radd(RExpr[t for t in f.terms if !(t isa RConst)]...), v.given, v.name),
                    "Var(X + c) = Var(X)")
        end
        ts = f.terms
        out = PExpr[Variance(t, v.given, v.name) for t in ts]
        for i in eachindex(ts), j in i+1:length(ts)
            push!(out, mul(Literal(2), Covariance(ts[i], ts[j], v.given, v.name)))
        end
        return (add(out...), "variance of a sum")
    end
    return nothing
end
rule_variance(::PExpr, ::SimplifyContext) = nothing

function rule_covariance(cv::Covariance, ctx::SimplifyContext)
    cv.f == cv.g && return (Variance(cv.f, cv.given, cv.name), "Cov(X, X) = Var(X)")
    known = _givenfree(cv)
    for x in (cv.f, cv.g)
        (x isa RConst || (!isempty(cv.given) && issubset(_rvars(x), known))) &&
            return (Literal(0), "covariance with a constant is 0")
    end
    a, fr = _rcoef(cv.f)
    b, gr = _rcoef(cv.g)
    (a != 1 || b != 1) && return (mul(Literal(a * b), Covariance(fr, gr, cv.given, cv.name)), "Cov(a X, b Y) = a b Cov(X, Y)")
    for (s, t, first_) in ((cv.f, cv.g, true), (cv.g, cv.f, false))
        s isa RAdd || continue
        return (add((Covariance(first_ ? x : t, first_ ? t : x, cv.given, cv.name) for x in s.terms)...),
                "covariance is bilinear")
    end
    any(a -> a isa Intervention, cv.given) && return nothing
    A = sort!(collect(_rvars(cv.f)))
    B = sort!(collect(_rvars(cv.g)))
    if isempty(intersect(A, B))
        why = _independent(ctx, cv.name, A, B, _rvsof(cv.given))
        why === nothing || return (Literal(0), "independent, so uncorrelated: " * why)
    end
    return nothing
end
rule_covariance(::PExpr, ::SimplifyContext) = nothing

function rule_power(p::Power, ::SimplifyContext)
    p.base isa Literal && return (Literal(p.base.value^p.p), "arithmetic")
    return nothing
end
rule_power(::PExpr, ::SimplifyContext) = nothing

push!(_RULES, rule_expect, rule_variance, rule_covariance, rule_power)

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

const _SUPERSCRIPT = Dict('0' => '⁰', '1' => '¹', '2' => '²', '3' => '³', '4' => '⁴', '5' => '⁵',
                          '6' => '⁶', '7' => '⁷', '8' => '⁸', '9' => '⁹', '-' => '⁻')
_sup(p) = p isa Integer ? map(c -> _SUPERSCRIPT[c], string(p)) : string("^", p)

_ratomic(x::Union{RVal,RConst,RInd,RFun,RQ}) = !(x isa RConst && x.value < 0)
_ratomic(::RExpr) = false

_rstr(x::RVal) = string(x.rv.name)
_rstr(x::RConst) = _numstr(x.value)
_rstr(x::RInd) = string("𝟙(", _astr(x.event), ")")
_rstr(x::RFun) = string(x.f, "(", _rstr(x.arg), ")")
_rstr(x::RQ) = sprint(show, x.e)
_rstr(x::RPow) = (_ratomic(x.base) ? _rstr(x.base) : "(" * _rstr(x.base) * ")") * _sup(x.p)
function _rstr(x::RMul)
    fs = x.factors
    s = ""
    for (i, f) in enumerate(fs)
        if i == 1 && f isa RConst
            s *= f.value == -1 ? "-" : _numstr(f.value)
            continue
        end
        part = f isa RAdd ? "(" * _rstr(f) * ")" : _rstr(f)
        s *= (s == "" || s == "-" || (i == 2 && fs[1] isa RConst && f isa RVal)) ? part : " " * part
    end
    return s
end
function _rstr(x::RAdd)
    s = ""
    for (i, t) in enumerate(x.terms)
        c, r = _rcoef(t)
        neg = c isa Real && c < 0
        body = neg ? _rstr(rmul(RConst(-c), r)) : _rstr(t)
        s *= i == 1 ? (neg ? "-" : "") * body : (neg ? " - " : " + ") * body
    end
    return s
end

_eletter(name::Symbol) = name === :P ? "E" : "E_$(name)"
_cond(given) = isempty(given) ? "" : " | " * _atomlist(given)

Base.show(io::IO, e::Expect) = print(io, _eletter(e.name), "[", _rstr(e.f), _cond(e.given), "]")
Base.show(io::IO, v::Variance) = print(io, v.name === :P ? "Var" : "Var_$(v.name)", "(", _rstr(v.f), _cond(v.given), ")")
Base.show(io::IO, c::Covariance) =
    print(io, c.name === :P ? "Cov" : "Cov_$(c.name)", "(", _rstr(c.f), ", ", _rstr(c.g), _cond(c.given), ")")
Base.show(io::IO, v::Value) = print(io, _ratomic(v.f) ? _rstr(v.f) : "(" * _rstr(v.f) * ")")
function Base.show(io::IO, p::Power)
    _atomic(p.base) ? show(io, p.base) : (print(io, "("); show(io, p.base); print(io, ")"))
    print(io, _sup(p.p))
end
Base.show(io::IO, x::RExpr) = print(io, _rstr(x))
Base.show(io::IO, g::GivenR) = print(io, _rstr(g.f), " | ", _atomlist(g.given))

_lxr(x::RVal) = _lx(x.rv.name)
_lxr(x::RConst) = _lx(x.value)
_lxr(x::RInd) = string("\\mathbb{1}_{", _lxatom(x.event), "}")
_lxr(x::RFun) = string(x.f === :abs ? "\\left|" : string(_lxfname(x.f), "\\left("), _lxr(x.arg),
                       x.f === :abs ? "\\right|" : "\\right)")
_lxr(x::RQ) = _latex(x.e)
_lxr(x::RPow) = string(_ratomic(x.base) ? _lxr(x.base) : "\\left(" * _lxr(x.base) * "\\right)", "^{", x.p, "}")
function _lxr(x::RMul)
    parts = String[]
    for (i, f) in enumerate(x.factors)
        if i == 1 && f isa RConst
            push!(parts, f.value == -1 ? "-" : _lx(f.value))
        else
            push!(parts, f isa RAdd ? "\\left(" * _lxr(f) * "\\right)" : _lxr(f))
        end
    end
    return join(parts, " ")
end
function _lxr(x::RAdd)
    s = ""
    for (i, t) in enumerate(x.terms)
        c, r = _rcoef(t)
        neg = c isa Real && c < 0
        body = neg ? _lxr(rmul(RConst(-c), r)) : _lxr(t)
        s *= i == 1 ? (neg ? "-" : "") * body : (neg ? " - " : " + ") * body
    end
    return s
end

_lxcond(given) = isempty(given) ? "" : " \\mid " * _lxatomlist(given)
_lxsubname(base, name) = name === :P ? base : string(base, "_{", _lx(name), "}")
_latex(e::Expect) = string(_lxsubname("\\mathbb{E}", e.name), "\\left[", _lxr(e.f), _lxcond(e.given), "\\right]")
_latex(v::Variance) = string(_lxsubname("\\operatorname{Var}", v.name), "\\left(", _lxr(v.f), _lxcond(v.given), "\\right)")
_latex(c::Covariance) = string(_lxsubname("\\operatorname{Cov}", c.name), "\\left(", _lxr(c.f), ", ",
                               _lxr(c.g), _lxcond(c.given), "\\right)")
_latex(v::Value) = _ratomic(v.f) ? _lxr(v.f) : "\\left(" * _lxr(v.f) * "\\right)"
_latex(p::Power) = string(_atomic(p.base) ? _latex(p.base) : "\\left(" * _latex(p.base) * "\\right)", "^{", p.p, "}")
_latex(x::RExpr) = _lxr(x)
latex(x::RExpr; env::Symbol=:inline) = _wrap(_lxr(x), env)

# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

_fmap(fn, F::Factor) = Factor(F.vars, F.domains, map(fn, F.values))

function _numeric_domain(bn::BayesNet, v::RV)
    d = domain(bn, v)
    all(x -> x isa Real, d) || throw(ArgumentError(
        "the values of $(v.name) ($(d)) are not numbers, so it has no expectation; give numeric `domains`"))
    return d
end

_rfactor(x::RVal, nets) = (d = _numeric_domain(_anynet(nets), x.rv); Factor(RV[x.rv], Vector{Any}[d], identity.(d)))

_rfactor(x::RConst, nets) = _scalar_factor(x.value)
_rfactor(x::RAdd, nets) = reduce(+, (_rfactor(t, nets) for t in x.terms))
_rfactor(x::RMul, nets) = reduce(*, (_rfactor(t, nets) for t in x.factors))
_rfactor(x::RPow, nets) = _fmap(y -> _numpow(y, x.p), _rfactor(x.base, nets))
_rfactor(x::RFun, nets) = _fmap(_rfunction(x.f), _rfactor(x.arg, nets))   # _rfunction: information.jl
_rfactor(x::RInd, nets) = _indicator(x.event, _anynet(nets))
_rfactor(x::RQ, nets) = _eval(x.e, nets)

function _eval(e::Expect, nets::_Nets)
    F = _rfactor(e.f, nets)
    targets = RV[v for v in F.vars if !(v in _givenfree(e))]
    isempty(targets) && return F
    Pf = _cond_factor(_net(nets, e.name), e.name, targets, Atom[], e.given)
    return sumout(F * Pf, targets)
end
function _eval(v::Variance, nets::_Nets)
    m = _eval(Expect(v.f, v.given, v.name), nets)
    return _eval(Expect(rpow(v.f, 2), v.given, v.name), nets) - _fmap(x -> x^2, m)
end
function _eval(c::Covariance, nets::_Nets)
    mf = _eval(Expect(c.f, c.given, c.name), nets)
    mg = _eval(Expect(c.g, c.given, c.name), nets)
    return _eval(Expect(rmul(c.f, c.g), c.given, c.name), nets) - mf * mg
end
_eval(v::Value, nets::_Nets) = _rfactor(v.f, nets)
_eval(p::Power, nets::_Nets) = _fmap(x -> x^p.p, _eval(p.base, nets))

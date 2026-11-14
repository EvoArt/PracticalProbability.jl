# Canonical forms and the algebra of random quantities.
#
# * `expand` writes a random quantity as a polynomial in its atoms (variables,
#   functions such as exp(X), indicators), with like terms collected and the
#   terms in a fixed order, so two equal polynomials print the same.
# * `simplify(r::RExpr)` brings a quotient of polynomials to lowest terms where
#   the denominator divides the numerator exactly (multivariate division).
# * `derivative` differentiates a random quantity (for moments from generating
#   functions, Jacobians and densities of transformed variables).
# * `canonical(e)` orders the arguments of every term in a probability
#   expression, so equal expressions print identically.
# * `simplify(e; search = true)` searches over the order in which rules are
#   applied for the smallest normal form, instead of taking the first match.

# ---------------------------------------------------------------------------
# min and max of random quantities
# ---------------------------------------------------------------------------

"""
    RExt(f, args)

`min(X, Y, …)` (`f = :min`) or `max(X, Y, …)` (`f = :max`) of random
quantities. Build with `min` and `max`.
"""
struct RExt <: RExpr
    f::Symbol
    args::Vector{RExpr}
end
Base.:(==)(a::RExt, b::RExt) = a.f === b.f && _multiset_eq(a.args, b.args)
Base.hash(a::RExt, h::UInt) = hash(_multiset_hash(a.args), hash(a.f, hash(:RExt, h)))

function _rext(f::Symbol, xs::RExpr...)
    args = RExpr[]
    for x in xs, y in (x isa RExt && x.f === f ? x.args : (x,))
        y in args || push!(args, y)
    end
    consts = [a.value for a in args if a isa RConst]
    if length(consts) == length(args)
        return RConst(f === :min ? minimum(consts) : maximum(consts))
    end
    length(args) == 1 && return args[1]
    return RExt(f, args)
end
for f in (:min, :max)
    @eval Base.$f(a::_R, b::_R) = _rext($(QuoteNode(f)), _r(a), _r(b))
    @eval Base.$f(a::_R, b::Number) = _rext($(QuoteNode(f)), _r(a), _r(b))
    @eval Base.$f(a::Number, b::_R) = _rext($(QuoteNode(f)), _r(a), _r(b))
end
_rvars(x::RExt) = mapreduce(_rvars, union, x.args; init=Set{RV}())
_ratomic(::RExt) = true
_rstr(x::RExt) = string(x.f, "(", join(map(_rstr, x.args), ", "), ")")
_lxr(x::RExt) = string("\\", x.f, "\\left(", join(map(_lxr, x.args), ", "), "\\right)")
_rfactor(x::RExt, nets) =
    reduce((a, b) -> _combine(x.f === :min ? min : max, a, b), (_rfactor(t, nets) for t in x.args))
_reval(x::RExt, vals) = (x.f === :min ? minimum : maximum)(_reval(t, vals) for t in x.args)

# Symbolic powers: `a^b` with a non-numeric exponent is exp(b log a).
Base.:^(a::_R, b::_R) = b isa RConst ? rpow(_r(a), b.value) : _rf(:exp, rmul(_r(b), _rf(:log, _r(a))))

Base.:^(a::Number, b::_R) = _rf(:exp, rmul(_r(b), RConst(log(a))))

# Other elementary functions of random quantities.
for f in (:atan, :sign)
    @eval Base.$f(x::_R) = RFun($(QuoteNode(f)), _r(x))
end

# ---------------------------------------------------------------------------
# Substitution
# ---------------------------------------------------------------------------

"""
    substitute(r::RExpr, X => value, …) -> RExpr

Replace variables in a random quantity by numbers, variables or other random
quantities, folding constants on the way: `substitute(exp(μ * t), t => 0)` is
`1`.
"""
substitute(r::RExpr, rules::Pair...) =
    _rsubs(r, Dict{RV,RExpr}(rv(p.first) => _r(p.second) for p in rules))
substitute(r::RV, rules::Pair...) = substitute(_r(r), rules...)

_rsubs(x::RVal, d) = get(d, x.rv, x)
_rsubs(x::RConst, d) = x
_rsubs(x::RAdd, d) = radd((_rsubs(t, d) for t in x.terms)...)
_rsubs(x::RMul, d) = rmul((_rsubs(t, d) for t in x.factors)...)
_rsubs(x::RPow, d) = rpow(_rsubs(x.base, d), x.p)
_rsubs(x::RFun, d) = _rf(x.f, _rsubs(x.arg, d))
_rsubs(x::RExt, d) = _rext(x.f, (_rsubs(t, d) for t in x.args)...)
_rsubs(x::RExpr, d) = x

# ---------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------

"""
    derivative(r, X) -> RExpr
    derivative(r, X, n)

The derivative of a random quantity with respect to the variable `X`
(`n`-th derivative with three arguments). Knows sums, products, powers,
`exp`, `log`, `sqrt`, `abs`, `atan`, `loggamma` and the standard normal CDF
`Φ`.
"""
derivative(r::RExpr, v::RV) = _d(r, v)
derivative(r::RV, v::RV) = _d(_r(r), v)
function derivative(r::Union{RExpr,RV}, v::RV, n::Integer)
    n >= 0 || throw(ArgumentError("the order of a derivative must be non-negative"))
    x = _r(r)
    for _ in 1:n
        x = _d(x, v)
    end
    return x
end

_d(x::RExpr, v::RV) = v in _rvars(x) ? _dd(x, v) : RConst(0)
_dd(x::RVal, v) = RConst(1)
_dd(x::RAdd, v) = radd((_d(t, v) for t in x.terms)...)
function _dd(x::RMul, v)
    fs = x.factors
    return radd((rmul(_d(fs[i], v), (fs[j] for j in eachindex(fs) if j != i)...) for i in eachindex(fs))...)
end
_dd(x::RPow, v) = rmul(RConst(x.p), rpow(x.base, x.p - 1), _d(x.base, v))
function _dd(x::RFun, v)
    a = x.arg
    da = _d(a, v)
    f = x.f
    g = f === :exp ? x :
        f === :log ? rpow(a, -1) :
        f === :sqrt ? rmul(RConst(1 // 2), rpow(a, -1 // 2)) :
        f === :abs ? _rf(:sign, a) :
        f === :atan ? rpow(radd(RConst(1), rpow(a, 2)), -1) :
        f === :loggamma ? _rf(:digamma, a) :
        f === :Phi ? rmul(RConst(1 / sqrt(2π)), _rf(:exp, rmul(RConst(-1 // 2), rpow(a, 2)))) :
        f === :sign ? RConst(0) :
        throw(ArgumentError(string("no derivative known for ", f)))
    return rmul(g, da)
end
_dd(x::RExpr, v) = throw(ArgumentError(string("cannot differentiate ", _rstr(x), " with respect to ", v.name)))

# ---------------------------------------------------------------------------
# Polynomials in atoms
# ---------------------------------------------------------------------------

# A monomial maps each atom (anything that is not a sum, product, constant or
# positive integer power) to its exponent; a polynomial maps monomials to
# coefficients.
const _Mono = Dict{RExpr,Int}
const _Poly = Dict{_Mono,Number}

_isnegligible(c) = c isa Union{Integer,Rational} ? iszero(c) : abs(c) < 1e-13

function _padd!(P::_Poly, m::_Mono, c)
    v = get(P, m, 0) + c
    _isnegligible(v) ? delete!(P, m) : (P[m] = v)
    return P
end
_pconst(c) = _isnegligible(c) ? _Poly() : _Poly(_Mono() => c)
function _pplus(P::_Poly, Q::_Poly)
    R = copy(P)
    for (m, c) in Q
        _padd!(R, m, c)
    end
    return R
end
_pscale(P::_Poly, c) = (R = _Poly(); for (m, d) in P; _padd!(R, m, c * d); end; R)
_mmul(a::_Mono, b::_Mono) = mergewith(+, a, b)
function _ptimes(P::_Poly, Q::_Poly)
    R = _Poly()
    for (m, c) in P, (n, d) in Q
        _padd!(R, _mmul(m, n), c * d)
    end
    return R
end
_ppow(P::_Poly, k::Integer) = k == 0 ? _pconst(1) : reduce(_ptimes, fill(P, k))

_poly(x::RConst) = _pconst(x.value)
_poly(x::RAdd) = reduce(_pplus, map(_poly, x.terms); init=_Poly())
_poly(x::RMul) = reduce(_ptimes, map(_poly, x.factors); init=_pconst(1))
function _poly(x::RPow)
    x.p isa Integer && x.p > 0 && return _ppow(_poly(x.base), x.p)
    return _patom(rpow(expand(x.base), x.p))
end
_poly(x::RFun) = _patom(_rf(x.f, expand(x.arg)))
_poly(x::RExt) = _patom(_rext(x.f, map(expand, x.args)...))
_poly(x::RExpr) = _patom(x)
_patom(a::RConst) = _pconst(a.value)
_patom(a::RExpr) = _Poly(_Mono(a => 1) => 1)

# Atoms in a fixed order: by printed form.
_akey(a::RExpr) = _rstr(a)
_sortedmono(m::_Mono) = sort!(collect(m); by=p -> _akey(p.first))
_degree(m::_Mono) = sum(values(m); init=0)
# Terms by decreasing degree, then by the printed form of their monomials.
_monokey(m::_Mono) = (-_degree(m), [(_akey(a), -e) for (a, e) in _sortedmono(m)])
_monoless(a::_Mono, b::_Mono) = isless(_monokey(a), _monokey(b))

function _unpoly(P::_Poly)
    isempty(P) && return RConst(0)
    ms = sort!(collect(keys(P)); lt=_monoless)
    terms = RExpr[]
    for m in ms
        c = P[m]
        fs = RExpr[rpow(a, e) for (a, e) in _sortedmono(m)]
        if isempty(fs)
            push!(terms, RConst(c))
        elseif isone(c)
            push!(terms, length(fs) == 1 ? fs[1] : RMul(fs))
        else
            push!(terms, RMul(vcat(RExpr[RConst(c)], fs)))
        end
    end
    return length(terms) == 1 ? terms[1] : RAdd(terms)
end

"""
    expand(r::RExpr) -> RExpr

Multiply out products and positive integer powers of sums and collect like
terms: the canonical polynomial form of a random quantity, with terms ordered
by degree and then alphabetically. `expand((X + Y)^2)` is `X² + 2 X Y + Y²`.
Function arguments (`exp(⋯)`, `log(⋯)`) are expanded too.
"""
expand(x::RExpr) = _unpoly(_poly(x))
expand(x::RV) = _r(x)

# Lexicographic order of monomials, with atoms ordered by printed form.
function _lexgreater(a::_Mono, b::_Mono, atoms)
    for x in atoms
        ea, eb = get(a, x, 0), get(b, x, 0)
        ea != eb && return ea > eb
    end
    return false
end
function _leading(P::_Poly, atoms)
    best = nothing
    for m in keys(P)
        (best === nothing || _lexgreater(m, best, atoms)) && (best = m)
    end
    return best
end

# N / D as a polynomial when D divides N exactly, otherwise `nothing`.
function _pdivide(N::_Poly, D::_Poly)
    isempty(D) && return nothing
    isempty(N) && return _Poly()
    atoms = sort!(unique(RExpr[a for P in (N, D) for m in keys(P) for a in keys(m)]); by=_akey)
    ld = _leading(D, atoms)
    lc = D[ld]
    Q = _Poly()
    R = copy(N)
    for _ in 1:10_000
        isempty(R) && return Q
        m = _leading(R, atoms)
        all(get(m, a, 0) >= e for (a, e) in ld) || return nothing
        qm = _Mono(a => e - get(ld, a, 0) for (a, e) in m if e - get(ld, a, 0) != 0)
        qc = _div(R[m], lc)
        _padd!(Q, qm, qc)
        R = _pplus(R, _pscale(_ptimes(_Poly(qm => 1), D), -qc))
    end
    return nothing
end

# A random quantity as numerator and denominator polynomials.
_ratform(x::RConst) = (_pconst(x.value), _pconst(1))
function _ratform(x::RAdd)
    N, D = _Poly(), _pconst(1)
    for t in x.terms
        n, d = _ratform(t)
        if d == D
            N = _pplus(N, n)
        else
            N = _pplus(_ptimes(N, d), _ptimes(n, D))
            D = _ptimes(D, d)
        end
    end
    return (N, D)
end
function _ratform(x::RMul)
    N, D = _pconst(1), _pconst(1)
    for f in x.factors
        n, d = _ratform(f)
        N, D = _ptimes(N, n), _ptimes(D, d)
    end
    return (N, D)
end
function _ratform(x::RPow)
    if x.p isa Integer
        n, d = _ratform(x.base)
        return x.p > 0 ? (_ppow(n, x.p), _ppow(d, x.p)) : (_ppow(d, -x.p), _ppow(n, -x.p))
    end
    return (_poly(x), _pconst(1))
end
_ratform(x::RExpr) = (_poly(x), _pconst(1))

"""
    simplify(r::RExpr) -> RExpr

A random quantity in lowest terms: written as one fraction of polynomials,
with the denominator divided out when it divides the numerator exactly
(multivariate polynomial division), and expanded. `simplify((a*b + a*c) / a)`
is `b + c`.
"""
function simplify(r::RExpr)
    N, D = _ratform(r)
    isempty(N) && return RConst(0)
    q = _pdivide(N, D)
    q === nothing || return _unpoly(q)
    # a constant denominator scales the numerator
    if length(D) == 1 && haskey(D, _Mono())
        return _unpoly(_pscale(N, _div(1, D[_Mono()])))
    end
    # cancel a common monomial factor, then keep N / D
    g = _commonmono(N, D)
    if !isempty(g)
        N = _pdivide(N, _Poly(g => 1))
        D = _pdivide(D, _Poly(g => 1))
    end
    return rmul(_unpoly(N), rpow(_unpoly(D), -1))
end
simplify(r::RV) = _r(r)

function _commonmono(P::_Poly, Q::_Poly)
    ms = vcat(collect(keys(P)), collect(keys(Q)))
    g = copy(ms[1])
    for m in ms[2:end]
        for a in collect(keys(g))
            g[a] = min(g[a], get(m, a, 0))
            g[a] == 0 && delete!(g, a)
        end
    end
    return g
end

# ---------------------------------------------------------------------------
# Canonical forms of probability expressions
# ---------------------------------------------------------------------------

_ekey(e) = sprint(show, e)
_sortatoms(xs) = sort(Atom[x for x in xs]; by=_astr)

"""
    canonical(e) -> same kind of expression

The canonical form of an expression: the arguments of every probability term
sorted, the factors of products and the terms of sums in a fixed order, and
random quantities expanded (see [`expand`](@ref)). Equal expressions (in the
sense of `==`, which already treats arguments as sets) have identical
canonical forms, so they print the same.
"""
canonical(p::Prob) = Prob(_sortatoms(p.targets), _sortatoms(p.given), p.name)
canonical(l::Literal) = l
function canonical(p::Product)
    fs = PExpr[canonical(f) for f in p.factors]
    c = [f for f in fs if f isa Literal]
    rest = sort!(PExpr[f for f in fs if !(f isa Literal)]; by=_ekey)
    return mul(c..., rest...)
end
function canonical(s::Sum)
    ts = PExpr[canonical(t) for t in s.terms]
    c = [t for t in ts if t isa Literal]
    rest = sort!(PExpr[t for t in ts if !(t isa Literal)]; by=t -> _ekey(_coef(t)[2]))
    return add(c..., rest...)
end
canonical(q::Quotient) = divide(canonical(q.num), canonical(q.den))
canonical(m::Marginal) = marg(sort(m.over), canonical(m.body))
canonical(e::Expect) = Expect(canonical(e.f), _sortatoms(e.given), e.name)
canonical(v::Variance) = Variance(canonical(v.f), _sortatoms(v.given), v.name)
function canonical(c::Covariance)
    f, g = canonical(c.f), canonical(c.g)
    _rstr(g) < _rstr(f) && ((f, g) = (g, f))
    return Covariance(f, g, _sortatoms(c.given), c.name)
end
canonical(v::Value) = Value(canonical(v.f))
canonical(p::Power) = canonical(p.base)^p.p
canonical(e::PExpr) = e
canonical(r::RExpr) = _canonr(expand(r))
canonical(r::RV) = _r(r)
canonical(eq::Equation) = Equation(canonical(eq.lhs), canonical(eq.rhs), eq.relation)
# Expectations inside random quantities get canonical forms too.
_canonr(x::RQ) = RQ(canonical(x.e))
_canonr(x::RAdd) = RAdd(RExpr[_canonr(t) for t in x.terms])
_canonr(x::RMul) = RMul(RExpr[_canonr(t) for t in x.factors])
_canonr(x::RPow) = RPow(_canonr(x.base), x.p)
_canonr(x::RExpr) = x

# ---------------------------------------------------------------------------
# simplify(…; search = true)
# ---------------------------------------------------------------------------

# Every single rewrite of `e`: each rule at the root, and each rewrite inside a child.
function _rewrite_all(e::PExpr, ctx::SimplifyContext)
    out = Tuple{PExpr,String}[]
    for r in _RULES
        o = r(e, ctx)
        o === nothing || push!(out, o)
    end
    cs = children(e)
    for (i, c) in enumerate(cs)
        for (ne, why) in _rewrite_all(c, ctx)
            newcs = copy(cs)
            newcs[i] = ne
            push!(out, (rebuild(e, newcs), why))
        end
    end
    return out
end

# Size of an expression: nodes, plus the arguments of each term.
_esize(p::Prob) = 1 + length(p.targets) + length(p.given)
_esize(::Literal) = 1
_esize(e::Union{Expect,Variance}) = 1 + _rsize(e.f) + length(e.given)
_esize(e::Covariance) = 1 + _rsize(e.f) + _rsize(e.g) + length(e.given)
_esize(v::Value) = _rsize(v.f)
_esize(e::PExpr) = 1 + sum(_esize, children(e); init=0)
_rsize(x::RAdd) = 1 + sum(_rsize, x.terms)
_rsize(x::RMul) = 1 + sum(_rsize, x.factors)
_rsize(x::RPow) = 1 + _rsize(x.base)
_rsize(x::RFun) = 1 + _rsize(x.arg)
_rsize(x::RQ) = _esize(x.e)
_rsize(::RExpr) = 1

# Beam search over rule orders for the smallest normal form (an expression no
# rule rewrites). The greedy first-match result is the fallback, kept on ties.
function _search_simplify(e::PExpr, ctx::SimplifyContext, greedy::Derivation, maxsteps::Integer, beam::Integer)
    best = greedy
    bestcost = isempty(_rewrite_all(last(greedy), ctx)) ? _esize(last(greedy)) : typemax(Int)
    seen = Set{String}([_ekey(canonical(e))])
    frontier = [derivation(e)]
    for _ in 1:maxsteps
        next = Derivation[]
        for d in frontier
            moves = _rewrite_all(last(d), ctx)
            if isempty(moves)
                c = _esize(last(d))
                c < bestcost && ((best, bestcost) = (d, c))
                continue
            end
            for (ne, why) in moves
                k = _ekey(canonical(ne))
                k in seen && continue
                push!(seen, k)
                nd = Derivation(copy(d.exprs), copy(d.rules))
                _push!(nd, ne, why)
                push!(next, nd)
            end
        end
        isempty(next) && break
        sort!(next; by=d -> _esize(last(d)))
        frontier = next[1:min(beam, length(next))]
    end
    return best
end

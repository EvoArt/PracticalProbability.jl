# Distributions of random variables and of functions of them.
#
# `laws(X => Family(:Normal, μ, σ), …)` says how variables are distributed.
# From that, closed forms: the law of a transformed variable (`law(2X + Y)`),
# moments, generating functions (MGF, PGF, characteristic function),
# densities and CDFs, the density of `g(X)` by a change of variables, and
# order statistics. `simplify(E(X^2); laws = L)` fills in moments.

# ---------------------------------------------------------------------------
# Laws
# ---------------------------------------------------------------------------

"""
    Laws

The distributions of some random variables, independent unless stated
otherwise. Build with [`laws`](@ref).
"""
struct Laws
    dists::Dict{RV,Family}
    independent::Bool
end

"""
    laws(X => Family(:Normal, μ, σ), Y => Family(:Poisson, λ); independent = true) -> Laws
    laws(X ~ Family(:Normal, μ, σ), …)

State the distributions of random variables, for [`law`](@ref),
[`MGF`](@ref) and friends, and for `simplify(…; laws)`. The variables are
mutually independent unless `independent = false`.
"""
laws(ps::Pair...; independent::Bool=true) =
    Laws(Dict{RV,Family}(rv(p.first) => p.second for p in ps), independent)
Base.:~(x::RV, f::Family) = x => f
Base.getindex(L::Laws, x::RV) = haskey(L.dists, x) ? L.dists[x] :
    throw(ArgumentError(string("no law given for ", x.name)))
Base.haskey(L::Laws, x::RV) = haskey(L.dists, x)
function Base.show(io::IO, L::Laws)
    print(io, "Laws(", join((string(k.name, " ~ ", sprint(show, v)) for (k, v) in sort!(collect(L.dists); by=first)), ", "),
          L.independent ? ", independent)" : ")")
end
simplify(e::PExpr, L::Laws; kwargs...) = simplify(e; laws=L, kwargs...)

_laws_independent(::Nothing, name, A, B) = false
_laws_independent(L::Laws, name, A, B) =
    name === :P && L.independent && isempty(intersect(A, B)) && all(v -> haskey(L, v), vcat(A, B))

# exact or symbolic arithmetic helpers
_rnum(x) = _r(x)
const _two = RConst(2)
const _half = RConst(1 // 2)
_rpowr(a::RExpr, b::RExpr) = a^b                    # canonical.jl: rpow for numbers, exp(b log a) otherwise
_isnum(x::RExpr, v) = x isa RConst && x.value == v
_isconst(x::RExpr) = x isa RConst
_isneg(x::RExpr) = x isa RConst && real(x.value) < 0

# Complex constants (characteristic functions) print with an `i`.
_numstr(x::Complex) = iszero(imag(x)) ? _numstr(real(x)) :
    iszero(real(x)) ? (isone(imag(x)) ? "i" : string(_numstr(imag(x)), "i")) : string(x)
_lx(x::Complex) = _numstr(x)

# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

"""
    E(f::Family) -> RExpr
    Var(f::Family) -> RExpr

The mean and variance of a distribution family, in its parameters:
`E(Family(:Gamma, α, β))` is `α β⁻¹`. See also [`rawmoment`](@ref),
[`centralmoment`](@ref) and [`cumulant`](@ref).
"""
E(f::Family) = _mean(f)
Var(f::Family) = _variance(f)

function _mean(f::Family)
    n, p = f.name, f.params
    n === :Normal && return p[1]
    n === :Laplace && return p[1]
    n === :Uniform && return rmul(_half, radd(p[1], p[2]))
    n === :Exponential && return rpow(p[1], -1)
    n === :Gamma && return _rdiv(p[1], p[2])
    n === :Beta && return _rdiv(p[1], radd(p[1], p[2]))
    n === :Poisson && return p[1]
    n === :Geometric && return rpow(p[1], -1)
    n === :Bernoulli && return p[1]
    n === :Binomial && return rmul(p[1], p[2])
    n === :LogNormal && return _rf(:exp, radd(p[1], rmul(_half, rpow(p[2], 2))))
    throw(ArgumentError(string(n, n === :Cauchy ? " has no mean" : " has no scalar mean here")))
end

function _variance(f::Family)
    n, p = f.name, f.params
    n === :Normal && return rpow(p[2], 2)
    n === :Laplace && return rmul(_two, rpow(p[2], 2))
    n === :Uniform && return rmul(RConst(1 // 12), rpow(_rsub(p[2], p[1]), 2))
    n === :Exponential && return rpow(p[1], -2)
    n === :Gamma && return _rdiv(p[1], rpow(p[2], 2))
    if n === :Beta
        a, b = p
        s = radd(a, b)
        return _rdiv(rmul(a, b), rmul(rpow(s, 2), radd(s, RConst(1))))
    end
    n === :Poisson && return p[1]
    n === :Geometric && return _rdiv(_rsub(RConst(1), p[1]), rpow(p[1], 2))
    n === :Bernoulli && return rmul(p[1], _rsub(RConst(1), p[1]))
    n === :Binomial && return rmul(p[1], p[2], _rsub(RConst(1), p[2]))
    if n === :LogNormal
        μ, σ = p
        return rmul(_rsub(_rf(:exp, rpow(σ, 2)), RConst(1)), _rf(:exp, radd(rmul(_two, μ), rpow(σ, 2))))
    end
    throw(ArgumentError(string(n, n === :Cauchy ? " has no variance" : " has no scalar variance here")))
end

const _T = RV(Symbol("t#"), :continuous)       # the argument of generating functions (a name users can't clash with)


"""
    rawmoment(f::Family, k) -> RExpr

`E[Xᵏ]` for `X ~ f`: closed forms where they are standard (Uniform,
LogNormal, Beta, Gamma, Exponential), otherwise the `k`-th derivative of the
moment generating function at zero. `rawmoment(Family(:Normal, μ, σ), 2)` is
`μ² + σ²`.
"""
function rawmoment(f::Family, k::Integer)
    k >= 0 || throw(ArgumentError("moments have a non-negative order"))
    k == 0 && return RConst(1)
    k == 1 && return _mean(f)
    n, p = f.name, f.params
    if n === :Uniform
        a, b = p
        return simplify(_rdiv(_rsub(rpow(b, k + 1), rpow(a, k + 1)), rmul(RConst(k + 1), _rsub(b, a))))
    elseif n === :LogNormal
        return _rf(:exp, radd(rmul(RConst(k), p[1]), rmul(RConst(k^2 // 2), rpow(p[2], 2))))
    elseif n === :Beta
        a, b = p
        return rmul((_rdiv(radd(a, RConst(r)), radd(a, b, RConst(r))) for r in 0:k-1)...)
    elseif n === :Gamma
        return rmul(rpow(p[2], -k), (radd(p[1], RConst(r)) for r in 0:k-1)...)
    elseif n === :Exponential
        return rmul(RConst(factorial(k)), rpow(p[1], -k))
    end
    return simplify(substitute(derivative(MGF(f, _T), _T, k), _T => 0))

end

"""
    centralmoment(f::Family, k) -> RExpr

`E[(X - E[X])ᵏ]` for `X ~ f`, from the raw moments.
"""
function centralmoment(f::Family, k::Integer)
    k == 0 && return RConst(1)
    k == 1 && return RConst(0)
    k == 2 && return _variance(f)
    μ = _mean(f)
    return simplify(radd((rmul(RConst(binomial(k, j)), rawmoment(f, j), rpow(rmul(RConst(-1), μ), k - j)) for j in 0:k)...))
end

"""
    cumulant(f::Family, k) -> RExpr

The `k`-th cumulant: the `k`-th derivative of the log of the moment
generating function at zero. `cumulant(Family(:Poisson, λ), k)` is `λ` for
every `k`.
"""
function cumulant(f::Family, k::Integer)
    k == 1 && return _mean(f)
    k == 2 && return _variance(f)
    return simplify(substitute(derivative(_rf(:log, MGF(f, _T)), _T, k), _T => 0))
end

# ---------------------------------------------------------------------------
# Generating functions
# ---------------------------------------------------------------------------

"""
    MGF(f::Family, t) -> RExpr
    MGF(r, L::Laws, t)

The moment generating function `E[exp(t X)]` of `X ~ f`, at `t` (a variable
or a number). With [`laws`](@ref), the MGF of a random quantity: for a linear
combination of independent variables, `M_{aX + bY + c}(t) = e^{ct} M_X(at) M_Y(bt)`.
Families without an MGF (Cauchy, LogNormal) raise an error; use [`CF`](@ref).
"""
MGF(f::Family, t) = _withargument(_mgf, f, t)
"""
    PGF(f::Family, s) -> RExpr
    PGF(r, L::Laws, s)

The probability generating function `E[sˣ]` of a distribution on the
non-negative integers (Bernoulli, Binomial, Poisson, Geometric). For sums of
independent variables the PGFs multiply.
"""
PGF(f::Family, s) = _withargument(_pgf, f, s)
"""
    CF(f::Family, t) -> RExpr
    CF(r, L::Laws, t)

The characteristic function `E[exp(i t X)]` (with `i` the imaginary unit).
It exists for every distribution, including the Cauchy (`exp(i x₀ t - γ |t|)`).
"""
CF(f::Family, t) = _withargument(_cf, f, t)

function _withargument(g, f::Family, t)
    t isa RV && return g(f, RVal(t))
    t isa RExpr && return g(f, t)
    return substitute(g(f, RVal(_T)), _T => t)
end

function _mgf(f::Family, t::RExpr)
    n, p = f.name, f.params
    if n === :Normal
        return _rf(:exp, radd(rmul(p[1], t), rmul(_half, rpow(p[2], 2), rpow(t, 2))))
    elseif n === :Laplace
        return _rdiv(_rf(:exp, rmul(p[1], t)), _rsub(RConst(1), rmul(rpow(p[2], 2), rpow(t, 2))))
    elseif n === :Uniform
        a, b = p
        return _rdiv(_rsub(_rf(:exp, rmul(b, t)), _rf(:exp, rmul(a, t))), rmul(t, _rsub(b, a)))
    elseif n === :Exponential
        return _rdiv(p[1], _rsub(p[1], t))
    elseif n === :Gamma
        return _rpowr(_rsub(RConst(1), _rdiv(t, p[2])), rmul(RConst(-1), p[1]))
    elseif n === :Poisson
        return _rf(:exp, rmul(p[1], _rsub(_rf(:exp, t), RConst(1))))
    elseif n === :Bernoulli
        return radd(_rsub(RConst(1), p[1]), rmul(p[1], _rf(:exp, t)))
    elseif n === :Binomial
        return _rpowr(radd(_rsub(RConst(1), p[2]), rmul(p[2], _rf(:exp, t))), p[1])
    elseif n === :Geometric
        q = p[1]
        return _rdiv(rmul(q, _rf(:exp, t)), _rsub(RConst(1), rmul(_rsub(RConst(1), q), _rf(:exp, t))))
    end
    throw(ArgumentError(string("no closed-form moment generating function for ", n,
                               n in (:Cauchy, :LogNormal) ? " (it does not exist; use CF)" : "")))
end

function _pgf(f::Family, s::RExpr)
    n, p = f.name, f.params
    n === :Poisson && return _rf(:exp, rmul(p[1], _rsub(s, RConst(1))))
    n === :Bernoulli && return radd(_rsub(RConst(1), p[1]), rmul(p[1], s))
    n === :Binomial && return _rpowr(radd(_rsub(RConst(1), p[2]), rmul(p[2], s)), p[1])
    n === :Geometric && return _rdiv(rmul(p[1], s), _rsub(RConst(1), rmul(_rsub(RConst(1), p[1]), s)))
    throw(ArgumentError(string("probability generating functions are for distributions on 0, 1, 2, …; got ", n)))
end

function _cf(f::Family, t::RExpr)
    if f.name === :Cauchy
        x0, γ = f.params
        return _rf(:exp, _rsub(rmul(RConst(im), x0, t), rmul(γ, _rf(:abs, t))))
    end
    f.name === :LogNormal && throw(ArgumentError("the LogNormal characteristic function has no closed form"))
    return _mgf(f, rmul(RConst(im), t))
end

# Generating functions of random quantities under laws.
for (G, g, mult) in ((:MGF, :_mgf, true), (:CF, :_cf, true), (:PGF, :_pgf, false))
    @eval function $G(r::Union{RV,RExpr}, L::Laws, t)
        x = _r(r)
        targ = t isa RV ? RVal(t) : t isa RExpr ? t : RVal(_T)
        out = _gf_under_laws($g, x, L, targ, $mult)
        return (t isa RV || t isa RExpr) ? out : substitute(out, _T => t)
    end
end

function _gf_under_laws(g, x::RExpr, L::Laws, t::RExpr, scaled::Bool)
    lin = _linear(x, L)
    if lin !== nothing && (L.independent || length(lin[1]) <= 1)
        coefs, b = lin
        if scaled
            parts = RExpr[g(L[v], rmul(a, t)) for (v, a) in coefs]
            # e^{bt} for an MGF; e^{ibt} for a CF
            shift = g === _cf ? rmul(RConst(im), b, t) : rmul(b, t)
            _isnum(b, 0) || push!(parts, _rf(:exp, shift))
            return rmul(parts...)
        else       # PGF of a sum of non-negative integer variables: coefficients must be 1
            all(a -> _isnum(a, 1), values(coefs)) && b isa RConst && isinteger(b.value) && b.value >= 0 ||
                throw(ArgumentError("a PGF of a combination needs a plain sum of variables plus a non-negative integer"))
            parts = RExpr[g(L[v], t) for v in keys(coefs)]
            _isnum(b, 0) || push!(parts, rpow(t, Int(b.value)))
            return rmul(parts...)
        end
    end
    return g(law(x, L), t)
end

# A random quantity as ∑ aᵢ Xᵢ + b, with the Xᵢ variables that have laws and
# the coefficients free of them, or `nothing`.
_linear(x::RExpr, L::Laws) = _linear_in(x, v -> haskey(L, v))
# The same for the variables `isvar` picks out.
function _linear_in(x::RExpr, isvar)
    P = _poly(x)
    coefs = Dict{RV,RExpr}()
    b = RExpr[]
    for (m, c) in P
        lawatoms = [a for a in keys(m) if a isa RVal && isvar(a.rv)]
        other = [a for a in keys(m) if !(a isa RVal && isvar(a.rv))]
        any(a -> any(isvar, _rvars(a)), other) && return nothing

        rest = rmul(RConst(c), (rpow(a, m[a]) for a in other)...)
        if isempty(lawatoms)
            push!(b, rest)
        elseif length(lawatoms) == 1 && m[lawatoms[1]] == 1
            v = lawatoms[1].rv
            coefs[v] = haskey(coefs, v) ? radd(coefs[v], rest) : rest
        else
            return nothing
        end
    end
    return (coefs, radd(b...))
end

# ---------------------------------------------------------------------------
# The law of a transformed variable
# ---------------------------------------------------------------------------

"""
    law(r, L::Laws) -> Family

The distribution of a random quantity, in closed form, from the laws of its
variables:

- affine maps `a X + b` of Normal, Uniform, Laplace and Cauchy variables, and
  scalings of Exponential, Gamma and LogNormal ones;
- sums of independent Normals, Cauchys, Poissons, Gammas (and Exponentials)
  with a common rate, and Binomials (and Bernoullis) with a common `p`;
- `exp` of a Normal (LogNormal) and `log` of a LogNormal (Normal);
- the square of a centred Normal (a Gamma: `σ² χ²₁`), so sums of squares of
  independent standard Normals are χ²;
- `min` of independent Exponentials (Exponential) and Geometrics (Geometric).

Anything else raises an `ArgumentError`. Positive symbolic coefficients are
assumed where a sign matters (`a X` for `X ~ Exponential(λ)`).
"""
law(r::RV, L::Laws) = L[r]
law(r::RExpr, L::Laws) = _law(expand(r), L)

_lawerr(x) = throw(ArgumentError(string("no closed form for the law of ", _rstr(x))))

_law(x::RVal, L::Laws) = L[x.rv]
_law(x::RConst, ::Laws) = _lawerr(x)
function _law(x::RAdd, L::Laws)
    consts = RExpr[t for t in x.terms if isempty(intersect(_rvars(t), keys(L.dists)))]
    rest = RExpr[t for t in x.terms if !(t in consts)]
    isempty(rest) && _lawerr(x)
    # group the terms by the variables they involve; groups must be disjoint
    groups = Vector{Vector{RExpr}}()
    gvars = Vector{Set{RV}}()
    for t in rest
        vs = Set(v for v in _rvars(t) if haskey(L, v))
        hit = [i for i in eachindex(groups) if !isempty(intersect(gvars[i], vs))]
        if isempty(hit)
            push!(groups, RExpr[t]); push!(gvars, vs)
        else
            i = hit[1]
            push!(groups[i], t); union!(gvars[i], vs)
            for j in reverse(hit[2:end])
                append!(groups[i], groups[j]); union!(gvars[i], gvars[j])
                deleteat!(groups, j); deleteat!(gvars, j)
            end
        end
    end
    length(groups) > 1 && !L.independent &&
        throw(ArgumentError("the law of a sum needs independent variables (laws(…; independent = true))"))
    length(groups) == 1 && length(groups[1]) > 1 && _lawerr(x)
    fams = Family[_law(radd(g...), L) for g in groups]
    f = reduce(_sumlaw, fams)
    b = radd(consts...)
    return _isnum(b, 0) ? f : _shift(f, b)
end
function _law(x::RMul, L::Laws)
    coef = RExpr[f for f in x.factors if isempty(intersect(_rvars(f), keys(L.dists)))]
    rest = RExpr[f for f in x.factors if !(f in coef)]
    isempty(coef) && _lawerr(x)
    inner = length(rest) == 1 ? rest[1] : RMul(rest)
    return _scale(_law(inner, L), rmul(coef...))
end
function _law(x::RFun, L::Laws)
    if x.f === :exp
        f = _law(x.arg, L)
        f.name === :Normal && return Family(:LogNormal, f.params...)
    elseif x.f === :log
        f = _law(x.arg, L)
        f.name === :LogNormal && return Family(:Normal, f.params...)
    end
    _lawerr(x)
end
function _law(x::RPow, L::Laws)
    if x.p == 2
        f = _law(x.base, L)
        if f.name === :Normal && _isnum(f.params[1], 0)       # σ² χ²₁ = Gamma(1/2, 1/(2σ²))
            return Family(:Gamma, RConst(1 // 2), rpow(rmul(_two, rpow(f.params[2], 2)), -1))
        end
    end
    _lawerr(x)
end
function _law(x::RExt, L::Laws)
    x.f === :min || _lawerr(x)
    L.independent || throw(ArgumentError("the law of a minimum needs independent variables"))
    all(a -> a isa RVal && haskey(L, a.rv), x.args) || _lawerr(x)
    fs = Family[L[a.rv] for a in x.args]
    if all(f -> f.name === :Exponential, fs)
        return Family(:Exponential, radd((f.params[1] for f in fs)...))
    elseif all(f -> f.name === :Geometric, fs)        # P(min > k) = ∏ (1 - pᵢ)ᵏ
        return Family(:Geometric, _rsub(RConst(1), rmul((_rsub(RConst(1), f.params[1]) for f in fs)...)))
    end
    _lawerr(x)
end
_law(x::RExpr, ::Laws) = _lawerr(x)

_absr(a::RExpr) = _rf(:abs, a)

function _scale(f::Family, a::RExpr)
    n, p = f.name, f.params
    n === :Normal && return Family(:Normal, rmul(a, p[1]), rmul(_absr(a), p[2]))
    n === :Laplace && return Family(:Laplace, rmul(a, p[1]), rmul(_absr(a), p[2]))
    n === :Cauchy && return Family(:Cauchy, rmul(a, p[1]), rmul(_absr(a), p[2]))
    if n === :Uniform
        lo, hi = rmul(a, p[1]), rmul(a, p[2])
        return _isneg(a) ? Family(:Uniform, hi, lo) : Family(:Uniform, lo, hi)
    end
    _isneg(a) && throw(ArgumentError(string("a negative multiple of a ", n, " variable has no law in the family")))
    n === :Exponential && return Family(:Exponential, _rdiv(p[1], a))
    n === :Gamma && return Family(:Gamma, p[1], _rdiv(p[2], a))
    n === :LogNormal && return Family(:LogNormal, radd(p[1], _rf(:log, a)), p[2])
    throw(ArgumentError(string("no closed form for a multiple of a ", n, " variable")))
end

function _shift(f::Family, b::RExpr)
    n, p = f.name, f.params
    n in (:Normal, :Laplace, :Cauchy) && return Family(n, radd(p[1], b), p[2])
    n === :Uniform && return Family(:Uniform, radd(p[1], b), radd(p[2], b))
    throw(ArgumentError(string("no closed form for a shifted ", n, " variable")))
end

_asgamma(f::Family) = f.name === :Exponential ? (RConst(1), f.params[1]) :
                      f.name === :Gamma ? (f.params[1], f.params[2]) : nothing
_asbinomial(f::Family) = f.name === :Bernoulli ? (RConst(1), f.params[1]) :
                         f.name === :Binomial ? (f.params[1], f.params[2]) : nothing

function _sumlaw(f::Family, g::Family)
    if f.name === g.name === :Normal
        return Family(:Normal, radd(f.params[1], g.params[1]),
                      _rf(:sqrt, radd(rpow(f.params[2], 2), rpow(g.params[2], 2))))
    elseif f.name === g.name === :Cauchy
        return Family(:Cauchy, radd(f.params[1], g.params[1]), radd(f.params[2], g.params[2]))
    elseif f.name === g.name === :Poisson
        return Family(:Poisson, radd(f.params[1], g.params[1]))
    end
    a, b = _asgamma(f), _asgamma(g)
    if a !== nothing && b !== nothing
        a[2] == b[2] && return Family(:Gamma, radd(a[1], b[1]), a[2])
        throw(ArgumentError("a sum of Gamma/Exponential variables has a Gamma law only with a common rate"))
    end
    a, b = _asbinomial(f), _asbinomial(g)
    if a !== nothing && b !== nothing
        a[2] == b[2] && return Family(:Binomial, radd(a[1], b[1]), a[2])
        throw(ArgumentError("a sum of Binomial/Bernoulli variables has a Binomial law only with a common p"))
    end
    throw(ArgumentError(string("no closed form for the law of a sum of ", f.name, " and ", g.name, " variables")))
end

# ---------------------------------------------------------------------------
# Densities and distribution functions
# ---------------------------------------------------------------------------

"""
    PDF(f::Family, x) -> RExpr

The density (continuous families) or probability mass function (discrete
ones) of `f` at `x`, a variable or a number, on the support of `f`.
"""
PDF(f::Family, x) = _withargument(_pdf, f, x)
"""
    CDF(f::Family, x) -> RExpr

The distribution function `P(X ≤ x)` on the support of `f`, for the families
where it is elementary or a normal CDF `Φ`: Normal, LogNormal, Exponential,
Uniform, Cauchy and Geometric.
"""
CDF(f::Family, x) = _withargument(_cdf, f, x)

const _SQRT2PI = RConst(sqrt(2π))
const _one = RConst(1)

function _pdf(f::Family, x::RExpr)
    n, p = f.name, f.params
    if n === :Normal
        μ, σ = p
        z = _rdiv(_rsub(x, μ), σ)
        return _rdiv(_rf(:exp, rmul(RConst(-1 // 2), rpow(z, 2))), rmul(σ, _SQRT2PI))
    elseif n === :LogNormal
        μ, σ = p
        z = _rdiv(_rsub(_rf(:log, x), μ), σ)
        return _rdiv(_rf(:exp, rmul(RConst(-1 // 2), rpow(z, 2))), rmul(x, σ, _SQRT2PI))
    elseif n === :Exponential
        return rmul(p[1], _rf(:exp, rmul(RConst(-1), p[1], x)))
    elseif n === :Gamma
        α, β = p
        return rmul(_rf(:exp, _rsub(rmul(α, _rf(:log, β)), _rlgamma(α))), _rpowr(x, _rsub(α, _one)),
                    _rf(:exp, rmul(RConst(-1), β, x)))
    elseif n === :Beta
        a, b = p
        return rmul(_rf(:exp, rmul(RConst(-1), _lbeta([a, b]))), _rpowr(x, _rsub(a, _one)),
                    _rpowr(_rsub(_one, x), _rsub(b, _one)))
    elseif n === :Uniform
        return rpow(_rsub(p[2], p[1]), -1)
    elseif n === :Laplace
        μ, b = p
        return _rdiv(_rf(:exp, rmul(RConst(-1), _rdiv(_rf(:abs, _rsub(x, μ)), b))), rmul(_two, b))
    elseif n === :Cauchy
        x0, γ = p
        return rpow(rmul(RConst(π), γ, radd(_one, rpow(_rdiv(_rsub(x, x0), γ), 2))), -1)
    elseif n === :Poisson
        λ = p[1]
        return _rf(:exp, radd(rmul(x, _rf(:log, λ)), rmul(RConst(-1), λ), rmul(RConst(-1), _rlgamma(radd(x, _one)))))
    elseif n === :Bernoulli
        q = p[1]
        return rmul(_rpowr(q, x), _rpowr(_rsub(_one, q), _rsub(_one, x)))
    elseif n === :Binomial
        m, q = p
        lchoose = radd(_rlgamma(radd(m, _one)), rmul(RConst(-1), _rlgamma(radd(x, _one))),
                       rmul(RConst(-1), _rlgamma(radd(_rsub(m, x), _one))))
        return rmul(_rf(:exp, lchoose), _rpowr(q, x), _rpowr(_rsub(_one, q), _rsub(m, x)))
    elseif n === :Geometric
        q = p[1]
        return rmul(q, _rpowr(_rsub(_one, q), _rsub(x, _one)))
    end
    throw(ArgumentError(string("no closed-form density for ", n)))
end

function _cdf(f::Family, x::RExpr)
    n, p = f.name, f.params
    n === :Normal && return _rf(:Phi, _rdiv(_rsub(x, p[1]), p[2]))
    n === :LogNormal && return _rf(:Phi, _rdiv(_rsub(_rf(:log, x), p[1]), p[2]))
    n === :Exponential && return _rsub(_one, _rf(:exp, rmul(RConst(-1), p[1], x)))
    n === :Uniform && return _rdiv(_rsub(x, p[1]), _rsub(p[2], p[1]))
    n === :Cauchy && return radd(RConst(1 // 2), rmul(RConst(1 / π), _rf(:atan, _rdiv(_rsub(x, p[1]), p[2]))))
    n === :Geometric && return _rsub(_one, _rpowr(_rsub(_one, p[1]), x))
    throw(ArgumentError(string("no elementary CDF for ", n)))
end

# ---------------------------------------------------------------------------
# Change of variables
# ---------------------------------------------------------------------------

"""
    change_of_variables(f, X, g, Y) -> RExpr

The density of `Y = g(X)` at `y`, where `X` has density `f` (a
[`Family`](@ref) or a random quantity in `X`) and `g` is one-to-one:
`f_X(g⁻¹(y)) |d g⁻¹(y) / dy|`. `g` is built from `+`, `*`, powers, `exp`,
`log` and `sqrt` with `X` appearing once; powers take the positive root.

```julia
@rv X::continuous Y::continuous
change_of_variables(Family(:Normal, 0, 1), X, exp(X), Y)    # the LogNormal(0, 1) density
```
"""
function change_of_variables(f, x::RV, g, y::RV)
    fx = f isa Family ? PDF(f, x) : _r(f)
    h = _invert(_r(g), x, RVal(y))
    return rmul(substitute(fx, x => h), _rf(:abs, derivative(h, y)))
end

"""
    jacobian(g, X, Y) -> RExpr

`|d g⁻¹(y) / dy|`, the factor a density picks up under `Y = g(X)` (see
[`change_of_variables`](@ref)). Also the Jacobian an importance weight needs
for a proposal in a transformed space.
"""
jacobian(g, x::RV, y::RV) = _rf(:abs, derivative(_invert(_r(g), x, RVal(y)), y))

_inverr(g, x) = throw(ArgumentError(string("cannot invert ", _rstr(g), " for ", x.name,
    ": give a one-to-one function of ", x.name, " built from +, *, powers, exp, log and sqrt, with ",
    x.name, " appearing once")))

# Solve g(x) = y for x.
function _invert(g::RExpr, x::RV, y::RExpr)
    g == RVal(x) && return y
    has(t) = x in _rvars(t)
    if g isa RAdd
        ts = [t for t in g.terms if has(t)]
        length(ts) == 1 || _inverr(g, x)
        rest = radd((t for t in g.terms if !has(t))...)
        return _invert(ts[1], x, _rsub(y, rest))
    elseif g isa RMul
        fs = [t for t in g.factors if has(t)]
        length(fs) == 1 || _inverr(g, x)
        rest = rmul((t for t in g.factors if !has(t))...)
        return _invert(fs[1], x, _rdiv(y, rest))
    elseif g isa RPow
        return _invert(g.base, x, rpow(y, g.p isa Union{Integer,Rational} ? 1 // g.p : 1 / g.p))
    elseif g isa RFun
        g.f === :exp && return _invert(g.arg, x, _rf(:log, y))
        g.f === :log && return _invert(g.arg, x, _rf(:exp, y))
        g.f === :sqrt && return _invert(g.arg, x, rpow(y, 2))
    end
    _inverr(g, x)
end

# ---------------------------------------------------------------------------
# Order statistics
# ---------------------------------------------------------------------------

"""
    order_statistic(f::Family, k, n, x) -> RExpr

The density at `x` of the `k`-th smallest of `n` independent draws from `f`:
`n! / ((k-1)! (n-k)!) F(x)^{k-1} (1 - F(x))^{n-k} f(x)`. `k = 1` is the
minimum and `k = n` the maximum; for `f` Exponential(λ) the minimum is
Exponential(nλ).
"""
function order_statistic(f::Family, k::Integer, n::Integer, x)
    1 <= k <= n || throw(ArgumentError("need 1 ≤ k ≤ n"))
    X = x isa RV ? RVal(x) : _r(x)
    F = CDF(f, X)
    c = factorial(big(n)) ÷ (factorial(big(k - 1)) * factorial(big(n - k)))
    c = c <= typemax(Int) ? Int(c) : c
    return rmul(RConst(c), rpow(F, k - 1), rpow(expand(_rsub(_one, F)), n - k), PDF(f, X))
end

# ---------------------------------------------------------------------------
# Moments in simplify
# ---------------------------------------------------------------------------

function rule_laws(e::Expect, ctx::SimplifyContext)
    L = ctx.laws
    (L isa Laws && e.name === :P && isempty(e.given)) || return nothing
    f = e.f
    if f isa RVal && haskey(L, f.rv)
        return (_value(_mean(L[f.rv])), string(f.rv.name, " ~ ", sprint(show, L[f.rv]), ": its mean"))
    elseif f isa RPow && f.base isa RVal && haskey(L, f.base.rv) && f.p isa Integer && f.p > 0
        d = L[f.base.rv]
        return (_value(rawmoment(d, f.p)), string(f.base.rv.name, " ~ ", sprint(show, d), ": its moment of order ", f.p))
    elseif f isa RFun && f.f === :exp
        lin = _linear(f.arg, L)
        if lin !== nothing && length(lin[1]) == 1 && isempty(intersect(_rvars(lin[2]), keys(L.dists)))
            v, a = first(lin[1])
            out = rmul(_rf(:exp, lin[2]), _mgf(L[v], a))
            return (_value(out), string("the moment generating function of ", sprint(show, L[v])))
        end
    end
    return nothing
end
function rule_laws(v::Variance, ctx::SimplifyContext)
    L = ctx.laws
    (L isa Laws && v.name === :P && isempty(v.given) && v.f isa RVal && haskey(L, v.f.rv)) || return nothing
    return (_value(_variance(L[v.f.rv])), string(v.f.rv.name, " ~ ", sprint(show, L[v.f.rv]), ": its variance"))
end
rule_laws(::PExpr, ::SimplifyContext) = nothing
push!(_RULES, rule_laws)

# A moment as a term of a probability expression (a number is a literal).
_value(r::RExpr) = r isa RConst ? Literal(r.value) : Value(r)

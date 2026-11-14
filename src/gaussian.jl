# Gaussian algebra: jointly Normal variables with symbolic means and
# covariances, their marginals and conditionals (the Schur complement), and
# linear-Gaussian structural equation models, whose covariances are sums over
# paths (Wright's path tracing) and whose causal effects are path products.

# ---------------------------------------------------------------------------
# Matrices of random quantities
# ---------------------------------------------------------------------------

_rmat(M::AbstractMatrix) = RExpr[_r(x) for x in M]
_rvec(v::AbstractVector) = RExpr[_r(x) for x in v]
_rmatmul(A::AbstractMatrix{RExpr}, B::AbstractMatrix{RExpr}) =
    RExpr[radd((rmul(A[i, k], B[k, j]) for k in axes(A, 2))...) for i in axes(A, 1), j in axes(B, 2)]
_rmatvec(A::AbstractMatrix{RExpr}, v::AbstractVector{RExpr}) =
    RExpr[radd((rmul(A[i, k], v[k]) for k in axes(A, 2))...) for i in axes(A, 1)]
_rmap(f, M::AbstractArray) = RExpr[f(x) for x in M]      # keeps the element type RExpr
_allconst(M) = all(x -> x isa RConst, M)

# Determinant by cofactor expansion (the matrices here are small).
function _rdet(M::AbstractMatrix{RExpr})
    n = size(M, 1)
    n == 0 && return RConst(1)
    n == 1 && return M[1, 1]
    n == 2 && return expand(_rsub(rmul(M[1, 1], M[2, 2]), rmul(M[1, 2], M[2, 1])))
    terms = RExpr[]
    for j in 1:n
        minor = M[2:end, [k for k in 1:n if k != j]]
        push!(terms, rmul(RConst(isodd(j) ? 1 : -1), M[1, j], _rdet(minor)))
    end
    return expand(radd(terms...))
end
# Adjugate: adj(M)[i, j] = (-1)^(i+j) det(M without row j and column i).
function _radj(M::AbstractMatrix{RExpr})
    n = size(M, 1)
    n == 1 && return reshape(RExpr[RConst(1)], 1, 1)
    return RExpr[rmul(RConst(isodd(i + j) ? -1 : 1),
                      _rdet(M[[k for k in 1:n if k != j], [k for k in 1:n if k != i]])) for i in 1:n, j in 1:n]
end

# ---------------------------------------------------------------------------
# Jointly Normal variables
# ---------------------------------------------------------------------------

"""
    Gaussian(vars, mean, cov)

Jointly Normal random variables `vars` with mean vector `mean` and covariance
matrix `cov`, whose entries are numbers, variables (symbolic parameters) or
random quantities. Marginals ([`marginal`](@ref)) and conditionals
([`conditional`](@ref)) are again `Gaussian`, computed symbolically;
[`law`](@ref) gives the Normal law of one variable and
[`regression`](@ref) the coefficients of a conditional mean.
"""
struct Gaussian
    vars::Vector{RV}
    mean::Vector{RExpr}
    cov::Matrix{RExpr}
    function Gaussian(vars::AbstractVector, μ::AbstractVector, Σ::AbstractMatrix)
        n = length(vars)
        (length(μ) == n && size(Σ) == (n, n)) ||
            throw(ArgumentError("a Gaussian over $(n) variables needs a mean of length $(n) and an $(n)×$(n) covariance"))
        S = _rmat(Σ)
        all(S[i, j] == S[j, i] || expand(_rsub(S[i, j], S[j, i])) == RConst(0) for i in 1:n, j in 1:n) ||
            throw(ArgumentError("the covariance matrix must be symmetric"))
        return new(RV[v for v in vars], _rvec(μ), S)
    end
end

function Base.show(io::IO, G::Gaussian)
    print(io, "Gaussian(", join((v.name for v in G.vars), ", "), ")")
end
function Base.show(io::IO, ::MIME"text/plain", G::Gaussian)
    println(io, "Gaussian over ", join((v.name for v in G.vars), ", "))
    println(io, "  mean: [", join(map(_rstr, G.mean), ", "), "]")
    print(io, "  cov:")
    for i in eachindex(G.vars)
        print(io, "\n    [", join(map(_rstr, G.cov[i, :]), ", "), "]")
    end
end

function _gidx(G::Gaussian, vs)
    idx = Int[]
    for v in vs
        k = findfirst(==(rv(v)), G.vars)
        k === nothing && throw(ArgumentError(string(rv(v).name, " is not a variable of the Gaussian")))
        push!(idx, k)
    end
    return idx
end

"""
    marginal(G::Gaussian, vars) -> Gaussian

The joint law of a subset of the variables: the matching entries of the mean
and covariance.
"""
function marginal(G::Gaussian, vars)
    k = _gidx(G, _aslist(vars))
    return Gaussian(G.vars[k], G.mean[k], G.cov[k, k])
end

"""
    conditional(G::Gaussian, targets, given) -> Gaussian

The law of `targets` given the variables `given`, by the Schur complement:
mean `μ_A + Σ_AB Σ_BB⁻¹ (x_B - μ_B)` and covariance
`Σ_AA - Σ_AB Σ_BB⁻¹ Σ_BA`. `given` lists variables (their values stay
symbolic, written as the variables themselves) or pairs `X => value`.
Entries are brought to lowest terms where the determinant of `Σ_BB` divides
them.
"""
function conditional(G::Gaussian, targets, given)
    gs = _aslist_pairs(given)
    A = _gidx(G, _aslist(targets))
    B = _gidx(G, [first(g) for g in gs])
    isempty(intersect(A, B)) || throw(ArgumentError("a variable cannot be both a target and conditioned on"))
    x = RExpr[g[2] for g in gs]
    SAB, SBB, SAA = G.cov[A, B], G.cov[B, B], G.cov[A, A]
    d = RExpr[_rsub(x[i], G.mean[B[i]]) for i in eachindex(B)]

    W, det = _rsolve_right(SAB, SBB)        # W * det = Σ_AB Σ_BB⁻¹ * det
    μ = RExpr[simplify(radd(G.mean[A[i]], _rdiv(_rmatvec(W, d)[i], det))) for i in eachindex(A)]
    C = _rmatmul(W, permutedims(SAB))
    Σ = RExpr[simplify(_rsub(SAA[i, j], _rdiv(C[i, j], det))) for i in eachindex(A), j in eachindex(A)]
    return Gaussian(G.vars[A], μ, Σ)
end

_aslist_pairs(x::Pair) = [(rv(x.first), _r(x.second))]
_aslist_pairs(x::RV) = [(x, RVal(x))]
_aslist_pairs(xs) = reduce(vcat, (_aslist_pairs(x) for x in xs); init=Tuple{RV,RExpr}[])

# Σ_AB Σ_BB⁻¹ as (numerator matrix, common denominator): adjugate and
# determinant symbolically, an exact inverse when everything is numeric.
function _rsolve_right(SAB::AbstractMatrix{RExpr}, SBB::AbstractMatrix{RExpr})
    if _allconst(SBB)
        inv = _exact_inv([x.value for x in SBB])
        return (_rmatmul(SAB, _rmat(inv)), RConst(1))
    end
    return (_rmatmul(SAB, _radj(SBB)), _rdet(SBB))
end

"""
    law(G::Gaussian, X) -> Family

The Normal law of one variable of a Gaussian.
"""
function law(G::Gaussian, x::RV)
    k = _gidx(G, [x])[1]
    v = G.cov[k, k]
    return Family(:Normal, G.mean[k], _rf(:sqrt, v))
end

"""
    regression(G::Gaussian, Y, Xs) -> (coefficients, intercept)

The coefficients of the conditional mean `E[Y | Xs] = intercept + ∑ βᵢ Xᵢ`
(the population least-squares regression of `Y` on `Xs`), simplified. For a
linear SEM with the back-door set among `Xs`, the coefficient of the treatment
is its causal effect.
"""
function regression(G::Gaussian, y::RV, xs)
    xs = _aslist(xs)
    c = conditional(G, [y], xs)
    m = c.mean[1]
    coefs = RExpr[simplify(derivative(m, rv(x))) for x in xs]
    intercept = simplify(substitute(m, (rv(x) => 0 for x in xs)...))
    return (coefs, intercept)
end

# ---------------------------------------------------------------------------
# Linear-Gaussian structural equation models
# ---------------------------------------------------------------------------

"""
    LinearSEM

A linear structural equation model with independent Gaussian noise:
`Xᵢ = cᵢ + ∑_{p ∈ pa(i)} b_{p→i} X_p + εᵢ`, `εᵢ ~ Normal(0, ωᵢ)` (ωᵢ a
variance). Build with [`linear_sem`](@ref).
"""
struct LinearSEM
    order::Vector{RV}
    parents::Dict{RV,Vector{RV}}
    coef::Dict{Tuple{RV,RV},RExpr}      # (parent, child) => coefficient
    intercept::Dict{RV,RExpr}
    noise::Dict{RV,RExpr}               # noise variances
end

"""
    linear_sem(X => rhs, …; noise = Dict()) -> LinearSEM

A linear-Gaussian SEM from its structural equations. Each right-hand side is a
linear combination of other variables of the model, with coefficients that
are numbers or symbolic parameters (any variable that is not a key), plus an
optional intercept; a root is `Z => 0`. `noise` gives each variable's noise
variance; by default `ω_X` (a symbolic parameter).

```julia
@rv X::continuous Y::continuous Z::continuous a::continuous b::continuous c::continuous
sem = linear_sem(Z => 0, X => a * Z, Y => b * X + c * Z)
total_effect(sem, X, Y)                 # b
regression(Gaussian(sem), Y, [X])       # b + a c ω_Z / (a² ω_Z + ω_X): confounded
regression(Gaussian(sem), Y, [X, Z])    # b, c: adjusting for Z
```
"""
function linear_sem(eqs::Pair...; noise::AbstractDict=Dict{RV,Any}())
    vars = RV[rv(e.first) for e in eqs]
    length(unique(vars)) == length(vars) || throw(ArgumentError("each variable needs exactly one equation"))
    isvar(v) = v in vars
    parents = Dict{RV,Vector{RV}}()
    coef = Dict{Tuple{RV,RV},RExpr}()
    intercept = Dict{RV,RExpr}()
    for (lhs, rhs) in eqs
        x = rv(lhs)
        lin = _linear_in(_r(rhs), isvar)
        lin === nothing && throw(ArgumentError(string("the equation for ", x.name,
            " must be linear in the model's variables with coefficients free of them")))
        cs, b = lin
        haskey(cs, x) && throw(ArgumentError(string(x.name, " cannot appear in its own equation")))
        parents[x] = sort!(collect(keys(cs)))
        for (p, a) in cs
            coef[(p, x)] = a
        end
        intercept[x] = b
    end
    order = topological_order(DAG(vars, parents))
    ω = Dict{RV,RExpr}(v => haskey(noise, v) ? _r(noise[v]) : RVal(RV(Symbol("ω_", v.name), :continuous)) for v in vars)
    return LinearSEM(order, parents, coef, intercept, ω)
end

dag(sem::LinearSEM) = DAG(sem.order, sem.parents)
Base.show(io::IO, sem::LinearSEM) = print(io, "LinearSEM(", join((v.name for v in sem.order), ", "), ")")
function Base.show(io::IO, ::MIME"text/plain", sem::LinearSEM)
    print(io, "Linear SEM with ", length(sem.order), " variables:")
    for v in sem.order
        rhs = radd(sem.intercept[v], (rmul(sem.coef[(p, v)], RVal(p)) for p in sem.parents[v])...)
        print(io, "\n  ", v.name, " = ", _isnum(rhs, 0) ? "" : _rstr(rhs) * " + ", "ε_", v.name,
              ",  Var(ε_", v.name, ") = ", _rstr(sem.noise[v]))
    end
end

# T[i, j]: the total effect of order[j] on order[i], the sum over directed
# paths of the products of the coefficients along them ((I - B)⁻¹).
function _total_effects(sem::LinearSEM)
    n = length(sem.order)
    pos = Dict(v => k for (k, v) in enumerate(sem.order))
    T = fill!(Matrix{RExpr}(undef, n, n), RConst(0))
    for (i, y) in enumerate(sem.order)
        T[i, i] = RConst(1)
        for p in sem.parents[y], j in 1:i-1
            T[i, j] = radd(T[i, j], rmul(sem.coef[(p, y)], T[pos[p], j]))
        end
    end
    return _rmap(expand, T)
end

"""
    total_effect(sem, X, Y) -> RExpr

The causal effect of `X` on `Y`, `∂E[Y | do(X = x)] / ∂x`: the sum over the
directed paths from `X` to `Y` of the products of their coefficients.
"""
function total_effect(sem::LinearSEM, x::RV, y::RV)
    T = _total_effects(sem)
    i, j = findfirst(==(y), sem.order), findfirst(==(x), sem.order)
    (i === nothing || j === nothing) && throw(ArgumentError("both variables must be in the model"))
    return T[i, j]
end

"""
    implied_covariance(sem) -> (vars, Σ)

The covariance matrix the SEM implies, `(I - B)⁻¹ Ω (I - B)⁻ᵀ`, entry by entry
a sum over treks (Wright's path tracing), in the variables' topological order.
"""
function implied_covariance(sem::LinearSEM)
    T = _total_effects(sem)
    n = length(sem.order)
    Ω = RExpr[i == j ? sem.noise[sem.order[i]] : RConst(0) for i in 1:n, j in 1:n]
    return (copy(sem.order), _rmap(expand, _rmatmul(_rmatmul(T, Ω), permutedims(T))))
end

"""
    Gaussian(sem::LinearSEM) -> Gaussian

The joint law of the variables of a linear-Gaussian SEM.
"""
function Gaussian(sem::LinearSEM)
    T = _total_effects(sem)
    vars, Σ = implied_covariance(sem)
    μ = _rmap(expand, _rmatvec(T, RExpr[sem.intercept[v] for v in vars]))
    return Gaussian(vars, μ, Σ)
end

"""
    intervene(sem, X => x, …) -> LinearSEM

The model under `do(X = x)`: `X`'s equation is replaced by the constant `x`
(a number or a symbol), with no noise. `Gaussian(intervene(sem, X => x))` is
the interventional distribution; its mean of `Y` is linear in `x` with slope
[`total_effect`](@ref).
"""
function intervene(sem::LinearSEM, fix::Pair...)
    parents = copy(sem.parents)
    coef = copy(sem.coef)
    intercept = copy(sem.intercept)
    noise = copy(sem.noise)
    for (x, val) in fix
        x = rv(x)
        x in sem.order || throw(ArgumentError(string(x.name, " is not in the model")))
        for p in parents[x]
            delete!(coef, (p, x))
        end
        parents[x] = RV[]
        intercept[x] = _r(val)
        noise[x] = RConst(0)
    end
    return LinearSEM(copy(sem.order), parents, coef, intercept, noise)
end

# ---------------------------------------------------------------------------
# The Kalman filter as Gaussian conditioning
# ---------------------------------------------------------------------------

"""
    kalman_step(; m, P, a, q, c, r, y) -> NamedTuple

One step of the scalar Kalman filter for `x_t = a x_{t-1} + w`, `w ~ N(0, q)`,
`y_t = c x_t + v`, `v ~ N(0, r)`, with `x_{t-1} ~ N(m, P)` from the previous
step, derived by conditioning the joint Gaussian of `(x_t, y_t)` on `y_t = y`.
Returns the prediction (`predicted_mean`, `predicted_var`), the `gain`, and the
filtered `mean` and `var`. Every argument may be a number or a symbol.
"""
function kalman_step(; m, P, a, q, c, r, y)
    x0, x1, y1 = RV(:x_prev, :continuous), RV(:x, :continuous), RV(:y_obs, :continuous)
    sem = LinearSEM([x0, x1, y1], Dict(x0 => RV[], x1 => [x0], y1 => [x1]),
                    Dict((x0, x1) => _r(a), (x1, y1) => _r(c)),
                    Dict(x0 => _r(m), x1 => RConst(0), y1 => RConst(0)),
                    Dict(x0 => _r(P), x1 => _r(q), y1 => _r(r)))
    G = marginal(Gaussian(sem), [x1, y1])
    post = conditional(G, [x1], [y1 => y])
    gain = simplify(_rdiv(G.cov[1, 2], G.cov[2, 2]))
    return (predicted_mean=G.mean[1], predicted_var=G.cov[1, 1], gain=gain,
            mean=post.mean[1], var=post.cov[1, 1])
end

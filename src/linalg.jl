# Exact linear algebra over any field (Rational, BigFloat, Float64, …) by
# Gauss–Jordan elimination. `deduce` and Markov chains use it so that rational
# inputs give rational answers; LinearAlgebra's `\`, `pinv` and `nullspace`
# work in floating point only.

_iszero_tol(x::Union{Integer,Rational}) = iszero(x)
_iszero_tol(x) = abs(x) < 1e-12

# Reduced row echelon form of `M` (copied). Returns the matrix and its pivot columns.
function _rref(M::AbstractMatrix)
    R = copy(M)
    m, n = size(R)
    pivots = Int[]
    r = 1
    for c in 1:n
        r > m && break
        # largest pivot for floats; first non-zero for exact types
        cand = [i for i in r:m if !_iszero_tol(R[i, c])]
        isempty(cand) && continue
        p = eltype(R) <: Union{Integer,Rational} ? cand[1] : cand[argmax([abs(R[i, c]) for i in cand])]
        if p != r
            R[[r, p], :] = R[[p, r], :]
        end
        piv = R[r, c]
        R[r, :] = R[r, :] ./ piv
        for i in 1:m
            i == r && continue
            f = R[i, c]
            _iszero_tol(f) || (R[i, :] = R[i, :] .- f .* R[r, :])
        end
        push!(pivots, c)
        r += 1
    end
    return R, pivots
end

# Every solution of A x = b, as a particular solution `x0` and a basis `N` of
# the null space (columns). `nothing` when there is no solution.
function _affine_solutions(A::AbstractMatrix, b::AbstractVector)
    T = promote_type(eltype(A), eltype(b))
    T <: Integer && (T = Rational{BigInt})
    T <: Rational && (T = Rational{BigInt})        # no overflow in elimination
    n = size(A, 2)
    R, pivots = _rref(hcat(T.(A), T.(b)))
    any(c -> c == n + 1, pivots) && return nothing   # a row 0 = 1
    x0 = zeros(T, n)
    for (k, c) in enumerate(pivots)
        x0[c] = R[k, n+1]
    end
    free = [c for c in 1:n if !(c in pivots)]
    N = zeros(T, n, length(free))
    for (j, f) in enumerate(free)
        N[f, j] = one(T)
        for (k, c) in enumerate(pivots)
            N[c, j] = -R[k, f]
        end
    end
    return x0, N
end

# The unique solution of a square system A x = b (exact for rationals).
function _exact_solve(A::AbstractMatrix, b::AbstractVector)
    s = _affine_solutions(A, b)
    s === nothing && throw(ArgumentError("the linear system has no solution"))
    x0, N = s
    size(N, 2) == 0 || throw(ArgumentError("the linear system has no unique solution"))
    return x0
end

# Inverse of a square matrix, exactly for rationals.
function _exact_inv(A::AbstractMatrix)
    n = size(A, 1)
    T = promote_type(eltype(A), Rational{Int})
    eltype(A) <: AbstractFloat && (T = eltype(A))
    R, pivots = _rref(hcat(T.(A), Matrix{T}(_eye(n))))
    length(pivots) == n && pivots == 1:n || throw(ArgumentError("the matrix is singular"))
    return R[:, n+1:end]
end
_eye(n) = [i == j ? 1 : 0 for i in 1:n, j in 1:n]


# Smallest exact type for a value: Rational{BigInt} → Rational{Int} when it fits.
_shrink(x::Rational{BigInt}) =
    (typemin(Int) <= numerator(x) <= typemax(Int) && denominator(x) <= typemax(Int)) ?
    Rational{Int}(x) : x
_shrink(x) = x

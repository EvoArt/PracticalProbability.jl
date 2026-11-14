# Finite discrete-time Markov chains, solved exactly: stationary
# distributions, communicating classes and periods, absorption (hitting)
# probabilities and expected hitting times. With a rational transition matrix
# every answer is an exact rational (gambler's ruin gives i/N, not 0.4999…).
# A chain unrolls into a BayesNet over X_1, …, X_n, so probability
# expressions about it compute like any other.

"""
    MarkovChain(P; states = 1:n)

A time-homogeneous Markov chain on finitely many states with transition
matrix `P`: `P[i, j]` is the probability of moving from state `i` to state
`j`, so every row sums to one. Integer and `Rational` matrices are kept
exact.

```julia
mc = MarkovChain([1//2 1//2; 1//4 3//4]; states = [:sun, :rain])
stationary(mc)                    # [1//3, 2//3]
```
"""
struct MarkovChain
    P::Matrix{<:Real}
    states::Vector{Any}
    function MarkovChain(P::AbstractMatrix; states=1:size(P, 1), atol::Real=1e-10)
        n = size(P, 1)
        size(P, 2) == n || throw(ArgumentError("a transition matrix is square"))
        length(states) == n || throw(ArgumentError("give one state name per row"))
        T = _tablearray(P)
        all(>=(0), T) || throw(ArgumentError("transition probabilities cannot be negative"))
        all(i -> isapprox(sum(T[i, :]), 1; atol), 1:n) ||
            throw(ArgumentError("every row of a transition matrix sums to one"))
        return new(T, collect(Any, states))
    end
end

Base.length(mc::MarkovChain) = length(mc.states)
states(mc::MarkovChain) = mc.states
transition_matrix(mc::MarkovChain) = mc.P
Base.show(io::IO, mc::MarkovChain) = print(io, "MarkovChain(", length(mc), " states)")
function Base.show(io::IO, ::MIME"text/plain", mc::MarkovChain)
    print(io, "Markov chain on ", length(mc), " states")
    for (i, s) in enumerate(mc.states)
        nz = [string(mc.states[j], " (", _numstr(mc.P[i, j]), ")") for j in eachindex(mc.states) if !iszero(mc.P[i, j])]
        print(io, "\n  ", s, " → ", join(nz, ", "))
    end
end

_sidx(mc::MarkovChain, s) = (k = findfirst(x -> isequal(x, s), mc.states);
    k === nothing ? throw(ArgumentError(string(repr(s), " is not a state of the chain"))) : k)
_sidxs(mc::MarkovChain, ss) = ss isa Union{AbstractVector,Tuple,AbstractSet} ? [_sidx(mc, s) for s in ss] : [_sidx(mc, ss)]

"""
    nstep(mc, n) -> Matrix

The `n`-step transition matrix `Pⁿ`.
"""
nstep(mc::MarkovChain, n::Integer) = mc.P^n

"""
    distribution(mc, init, n) -> Vector

The distribution of `X_{n+1}` when `X_1` has distribution `init` (a vector
over the states), `init ⋅ Pⁿ`.
"""
distribution(mc::MarkovChain, init::AbstractVector, n::Integer) = vec(permutedims(_tablearray(init)) * mc.P^n)

# Which states each state reaches (in any number of steps, including zero).
function _reach(mc::MarkovChain)
    n = length(mc)
    R = [i == j || !iszero(mc.P[i, j]) for i in 1:n, j in 1:n]
    for k in 1:n, i in 1:n, j in 1:n
        R[i, j] |= R[i, k] && R[k, j]
    end
    return R
end

"""
    communicating_classes(mc) -> Vector of NamedTuples

The communicating classes of the chain, each with its `states`, whether it is
`recurrent` (closed: the chain never leaves it) or transient, and its
`period`.
"""
function communicating_classes(mc::MarkovChain)
    R = _reach(mc)
    n = length(mc)
    seen = falses(n)
    out = NamedTuple{(:states, :recurrent, :period),Tuple{Vector{Any},Bool,Int}}[]
    for i in 1:n
        seen[i] && continue
        cls = [j for j in 1:n if R[i, j] && R[j, i]]
        seen[cls] .= true
        closed = all(!R[c, j] || j in cls for c in cls, j in 1:n)
        push!(out, (states=mc.states[cls], recurrent=closed, period=_period(mc, cls)))
    end
    return out
end

# gcd of the cycle lengths in a class, from breadth-first levels.
function _period(mc::MarkovChain, cls::Vector{Int})
    level = Dict(cls[1] => 0)
    queue = [cls[1]]
    while !isempty(queue)
        u = popfirst!(queue)
        for v in cls
            if !iszero(mc.P[u, v]) && !haskey(level, v)
                level[v] = level[u] + 1
                push!(queue, v)
            end
        end
    end
    g = 0
    for u in cls, v in cls
        iszero(mc.P[u, v]) || (g = gcd(g, level[u] + 1 - level[v]))
    end
    return g == 0 ? 0 : abs(g)
end

"""
    isirreducible(mc) -> Bool
    isaperiodic(mc) -> Bool

Whether every state reaches every other, and whether every recurrent class
has period one.
"""
isirreducible(mc::MarkovChain) = all(_reach(mc))
isaperiodic(mc::MarkovChain) = all(c -> !c.recurrent || c.period == 1, communicating_classes(mc))

"""
    stationary(mc) -> Vector

The stationary distribution `π = π P` with `∑ π = 1`, exactly for rational
chains. A chain with several recurrent classes has many; then this is an
error, and [`communicating_classes`](@ref) shows the classes.
"""
function stationary(mc::MarkovChain)
    n = length(mc)
    A = vcat(permutedims(mc.P) - _eye(n), ones(Int, 1, n))
    b = vcat(zeros(Int, n), 1)
    sol = _affine_solutions(A, b)
    sol === nothing && throw(ArgumentError("no stationary distribution (this cannot happen for a finite chain)"))
    x0, N = sol
    size(N, 2) == 0 || throw(ArgumentError(
        "the stationary distribution is not unique: the chain has several recurrent classes"))
    return _shrink.(x0)
end

"""
    hitting_probability(mc, target) -> Vector

For each starting state, the probability that the chain ever enters `target`
(a state or a collection of states): `1` on the target, and for absorbing
chains the absorption probabilities (gambler's ruin: `i / N`).
"""
function hitting_probability(mc::MarkovChain, target)
    n = length(mc)
    tgt = _sidxs(mc, target)
    R = _reach(mc)
    can = [any(R[i, t] for t in tgt) for i in 1:n]
    free = [i for i in 1:n if can[i] && !(i in tgt)]
    h = Vector{Any}(undef, n)
    for i in 1:n
        h[i] = i in tgt ? 1 : (can[i] ? nothing : 0)
    end
    if !isempty(free)
        # h_i - ∑_{j free} P_ij h_j = ∑_{j ∈ target} P_ij
        A = [(i == j ? 1 : 0) - mc.P[i, j] for i in free, j in free]
        b = [sum(mc.P[i, t] for t in tgt) for i in free]
        x = _exact_solve(A, b)
        for (k, i) in enumerate(free)
            h[i] = _shrink(x[k])
        end
    end
    return identity.(h)
end

"""
    hitting_time(mc, target) -> Vector

For each starting state, the expected number of steps until the chain first
enters `target`: `0` on the target, `Inf` where it might never get there.
"""
function hitting_time(mc::MarkovChain, target)
    n = length(mc)
    tgt = _sidxs(mc, target)
    h = hitting_probability(mc, target)
    sure = [i for i in 1:n if !(i in tgt) && (h[i] isa _Exact ? isone(h[i]) : isapprox(h[i], 1; atol=1e-10))]

    k = Vector{Any}(undef, n)
    for i in 1:n
        k[i] = i in tgt ? 0 : (i in sure ? nothing : Inf)
    end
    if !isempty(sure)
        # k_i - ∑_{j sure} P_ij k_j = 1   (states that may miss the target are never entered from a sure one)
        A = [(i == j ? 1 : 0) - mc.P[i, j] for i in sure, j in sure]
        x = _exact_solve(A, ones(Int, length(sure)))
        for (m, i) in enumerate(sure)
            k[i] = _shrink(x[m])
        end
    end
    return identity.(k)
end

"""
    mean_return_time(mc, s) -> Real

The expected number of steps to return to `s` starting from `s`; `1 / π_s`
for an irreducible chain.
"""
function mean_return_time(mc::MarkovChain, s)
    i = _sidx(mc, s)
    k = hitting_time(mc, s)
    return _shrink(1 + sum(mc.P[i, j] * k[j] for j in eachindex(mc.states) if j != i; init=0))
end

"""
    BayesNet(mc::MarkovChain, X, n; init) -> BayesNet

The first `n` steps of the chain as a Bayesian network over `X[1], …, X[n]`
(the variables `X_1, …, X_n`), with `X[1] ~ init` and `X[t+1] | X[t]` from
the transition matrix. Probability expressions about the chain then compute
exactly: `compute(P(X[3](2) | X[1](1)), bn)` is `P²[1, 2]`.
"""
function BayesNet(mc::MarkovChain, x::RV, n::Integer; init::AbstractVector)
    n >= 1 || throw(ArgumentError("a chain needs at least one step"))
    length(init) == length(mc) || throw(ArgumentError("init needs one probability per state"))
    xs = [x[t] for t in 1:n]
    specs = Pair[xs[1] => _tablearray(init)]
    for t in 2:n
        push!(specs, Given(Atom[xs[t]], Atom[xs[t-1]]) => permutedims(mc.P))   # rows: X_t, columns: X_{t-1}
    end
    return BayesNet(specs...; domains=Dict(v => mc.states for v in xs))
end

# Experiments: build a Bayesian network from a description of a chance
# experiment (dice, coins, urns, cards, "the host opens a door you didn't
# pick") instead of writing probability tables by hand. Equally likely
# outcomes get exact rational probabilities, so answers come out as 1//6 and
# 2//3.

"""
    Outcomes(values, probs)

A distribution over a list of outcomes: one step of an experiment. Build with
[`uniform`](@ref), [`die`](@ref), [`coin`](@ref), [`bernoulli`](@ref) and
[`categorical`](@ref).
"""
struct Outcomes
    values::Vector{Any}
    probs::Vector{Real}
    function Outcomes(values, probs)
        length(values) == length(probs) || throw(ArgumentError("one probability per outcome"))
        all(>=(0), probs) || throw(ArgumentError("probabilities cannot be negative"))
        isapprox(sum(probs), 1; atol=1e-10) || throw(ArgumentError("the probabilities must sum to one"))
        # merge repeated outcomes
        vs, ps = Any[], Real[]
        for (v, p) in zip(values, probs)
            k = findfirst(x -> isequal(x, v), vs)
            k === nothing ? (push!(vs, v); push!(ps, p)) : (ps[k] += p)
        end
        return new(vs, ps)
    end
end

"""
    uniform(values)

Each of `values` equally likely, with exact rational probabilities.
"""
function uniform(values)
    vs = collect(Any, values)
    isempty(vs) && throw(ArgumentError("uniform needs at least one outcome"))
    return Outcomes(vs, fill(1 // length(vs), length(vs)))
end
"""
    die(n = 6)

A fair `n`-sided die: `1:n`, each with probability `1//n`.
"""
die(n::Integer=6) = uniform(1:n)
"""
    coin(p = 1//2)

A coin landing `:heads` with probability `p`.
"""
coin(p::Real=1 // 2) = Outcomes(Any[:heads, :tails], Real[p, 1 - p])
"""
    bernoulli(p)

`true` with probability `p`, `false` otherwise: the step for a named event
(`@event A`).
"""
bernoulli(p::Real) = Outcomes(Any[true, false], Real[p, 1 - p])
"""
    categorical(value => prob, …)

Given outcomes with given probabilities.
"""
categorical(ps::Pair...) = Outcomes(Any[p.first for p in ps], Real[p.second for p in ps])

"""
    Depends(parents, f, deterministic)

A step whose outcome depends on earlier ones: `f(parent values...)` returns
the [`Outcomes`](@ref) (or, for [`derived`](@ref), the value). Build with
[`depends_on`](@ref) or [`derived`](@ref).
"""
struct Depends
    parents::Vector{RV}
    f::Any
    deterministic::Bool
end

"""
    depends_on(f, X, Y, …)

A step whose distribution depends on the outcomes of `X, Y, …` (declared
earlier in the experiment): `f(x, y, …)` returns its [`Outcomes`](@ref).
The Monty Hall host: `depends_on((car, pick) -> uniform(setdiff(1:3, [car, pick])), Car, Pick)`.
"""
depends_on(f, parents::RV...) = Depends(RV[parents...], f, false)

"""
    derived(f, X, Y, …)

A variable computed from earlier outcomes: `f(x, y, …)` is its value. The
sum of two dice is `derived(+, D1, D2)`; a named event can be derived from a
condition, `derived((a, b) -> a == b, D1, D2)`.
"""
derived(f, parents::RV...) = Depends(RV[parents...], f, true)

"""
    urn(:red => 3, :blue => 2)

An urn holding balls of the given colours (any values) and counts, for
[`draws`](@ref).
"""
struct Urn
    counts::Vector{Pair{Any,Int}}
end
urn(ps::Pair...) = Urn(Pair{Any,Int}[p.first => Int(p.second) for p in ps])

"""
    deck()

A standard 52-card deck as an [`Urn`](@ref) of `(rank, suit)` cards, ranks
`2:10`, `:J`, `:Q`, `:K`, `:A` and suits `:♠, :♥, :♦, :♣`. Each draw from it
has 52 outcomes, so a network of more than about three draws gets large; draw
from an urn of what matters (`urn(:ace => 4, :other => 48)`) instead.
"""
deck() = Urn(Pair{Any,Int}[(r, s) => 1 for s in (:♠, :♥, :♦, :♣) for r in Any[2:10..., :J, :Q, :K, :A]])

"""
    draws(u::Urn, X1, X2, …; replace = false) -> Vector

Successive draws from an urn, without replacement unless `replace = true`:
the steps for an [`experiment`](@ref). Without replacement each draw depends
on all earlier ones.
"""
function draws(u::Urn, xs::RV...; replace::Bool=false)
    steps = Pair{RV,Any}[]
    for (k, x) in enumerate(xs)
        prev = RV[xs[1:k-1]...]
        if replace || k == 1
            push!(steps, x => _urnoutcomes(u.counts, ()))
        else
            push!(steps, x => Depends(prev, (vals...) -> _urnoutcomes(u.counts, vals), false))
        end
    end
    return steps
end
function _urnoutcomes(counts, taken)
    left = [(v, c - count(t -> isequal(t, v), taken)) for (v, c) in counts]
    total = sum(last, left)
    total > 0 || throw(ArgumentError("the urn is empty"))
    keep = [(v, c) for (v, c) in left if c > 0]
    return Outcomes(Any[v for (v, _) in keep], Real[c // total for (_, c) in keep])
end

"""
    experiment(X => step, …) -> BayesNet

A Bayesian network from a description of an experiment, one variable per
step, in order. A step is [`Outcomes`](@ref) (`die()`, `coin()`,
`uniform(…)`, `categorical(…)`), [`depends_on`](@ref) earlier variables, or
[`derived`](@ref) from them; [`draws`](@ref) gives several steps at once.
Probabilities stay exact where they are rational.

```julia
@rv D1 D2 S
@event Doubles
bn = experiment(D1 => die(), D2 => die(), S => derived(+, D1, D2),
                Doubles => derived(==, D1, D2))
compute(P(S(7)), bn)              # 1//6
compute(P(Doubles | S(8)), bn)    # 1//5

@rv Car Pick Host
monty = experiment(Car => uniform(1:3), Pick => uniform(1:3),
                   Host => depends_on((c, p) -> uniform(setdiff(1:3, [c, p])), Car, Pick))
compute(P(Car(2) | Pick(1), Host(3)), monty)    # 2//3: switch
```
"""
function experiment(steps...)
    flat = Pair[]
    for s in steps
        s isa Pair ? push!(flat, s) : append!(flat, s)
    end
    doms = Dict{RV,Vector{Any}}()
    specs = Pair[]
    for (lhs, step) in flat
        x = rv(lhs)
        haskey(doms, x) && throw(ArgumentError(string(x.name, " appears twice in the experiment")))
        if step isa Outcomes
            dom, T = _stepdomain(x, [step]), nothing
            T = [_probof(step, v) for v in dom]
            push!(specs, x => T)
        elseif step isa Depends
            for p in step.parents
                haskey(doms, p) || throw(ArgumentError(string(x.name, " depends on ", p.name,
                                                              ", which must come earlier in the experiment")))
            end
            pdoms = [doms[p] for p in step.parents]
            cells = CartesianIndices(Tuple(length.(pdoms)))
            outs = Dict{CartesianIndex,Outcomes}()
            for c in cells
                vals = [pdoms[k][c[k]] for k in eachindex(pdoms)]
                r = step.f(vals...)
                outs[c] = step.deterministic ? Outcomes(Any[r], Real[1]) :
                          r isa Outcomes ? r :
                          throw(ArgumentError(string("the step for ", x.name, " must return Outcomes (uniform, categorical, …)")))
            end
            dom = _stepdomain(x, collect(values(outs)))
            T = Array{Real}(undef, length(dom), size(cells)...)
            for c in cells, (i, v) in enumerate(dom)
                T[i, Tuple(c)...] = _probof(outs[c], v)
            end
            push!(specs, Given(Atom[x], Atom[step.parents...]) => (isempty(step.parents) ? T[:] : T))
        else
            throw(ArgumentError(string("a step is Outcomes, depends_on(…) or derived(…); got ", repr(step))))
        end
        doms[x] = dom
    end
    return BayesNet(specs...; domains=doms)
end

_probof(o::Outcomes, v) = (k = findfirst(x -> isequal(x, v), o.values); k === nothing ? 0 : o.probs[k])

# The values a variable can take, in a stable order (sorted when possible).
function _stepdomain(x::RV, outs)
    vs = Any[]
    for o in outs, v in o.values
        any(y -> isequal(y, v), vs) || push!(vs, v)
    end
    if isevent(x)
        all(v -> v isa Bool, vs) || throw(ArgumentError(string("the event ", x.name, " needs true/false outcomes")))
        return Any[true, false]
    end
    try
        sort!(vs)
    catch
    end
    return vs
end

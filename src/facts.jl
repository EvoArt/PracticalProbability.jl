# Facts: what is known about events and variables without a graph.
# Independence (`A ⟂ B | C`), disjointness, implication, and known
# probabilities. `simplify(…; assume = facts)` rewrites with the structural
# facts; `deduce(query, facts)` uses all of them, including the numbers.
# (The rules are in assume.jl; the types are here because SimplifyContext
# holds a `Facts`.)

"""
    disjoint(A, B, …)

The fact that the events are mutually exclusive (pairwise disjoint):
`P(A ∩ B) = 0`, and `P(A ∪ B) = P(A) + P(B)`. Pass it to [`assume`](@ref).
"""
struct Disjoint
    parts::Vector{Atom}
end
disjoint(xs::Atom...) = Disjoint(Atom[_check_event(x) for x in xs])

"""
    implies(A, B)

The fact that `A` implies `B` (`A ⊆ B`): `P(A ∩ B) = P(A)`,
`P(B | A) = 1`, `P(A ∪ B) = P(B)`. Pass it to [`assume`](@ref).
"""
struct Implies
    a::Atom
    b::Atom
end
implies(a::Atom, b::Atom) = Implies(_check_event(a), _check_event(b))

"""
    iid(y, given...)

The fact that the elements `y[i]` of an indexed variable are independent and
identically distributed given `given` (e.g. `iid(y, θ)`):
`P(y[i] | y[others], θ) = P(y[i] | θ)` and `P(y[1:n] | θ) = ∏ᵢ P(y[i] | θ)`.
"""
struct IID
    base::RV
    given::Vector{Atom}
end
iid(y::RV, given::Atom...) = IID(y, Atom[given...])

"""
    markov(x)

The (first-order) Markov property of a sequence `x[t]`:
`P(x[t] | x[1:t-1], C) = P(x[t] | x[t-1], C)`.
"""
struct Markov
    base::RV
end
markov(x::RV) = Markov(x)

"""
    Facts
    assume(facts...) -> Facts

A set of assumptions: [`Independence`](@ref) statements (`A ⟂ B`,
`A ⟂ B | C`), [`disjoint`](@ref) and [`implies`](@ref) facts, [`iid`](@ref)
and [`markov`](@ref) for indexed variables, and known values written as
pairs, `P(A) => 0.3`, `P(B | A) => 0.9`, `P(A ∪ B) => 0.6`.

Use with `simplify(expr; assume = facts)`, which applies the structural facts,
and [`deduce`](@ref), which also uses the numbers.
"""
struct Facts
    independent::Vector{Independence}
    disjoint::Vector{Disjoint}
    implies::Vector{Implies}
    values::Vector{Pair{PExpr,Real}}
    iid::Vector{IID}
    markov::Vector{Markov}
end
Facts() = Facts(Independence[], Disjoint[], Implies[], Pair{PExpr,Real}[], IID[], Markov[])

function assume(xs...)
    f = Facts()
    foreach(x -> _addfact!(f, x), xs)
    return f
end

_addfact!(f::Facts, x::Independence) = push!(f.independent, x)
_addfact!(f::Facts, x::Disjoint) = push!(f.disjoint, x)
_addfact!(f::Facts, x::Implies) = push!(f.implies, x)
_addfact!(f::Facts, x::Pair{<:PExpr,<:Real}) = push!(f.values, x.first => _num(x.second))

_addfact!(f::Facts, x::IID) = push!(f.iid, x)
_addfact!(f::Facts, x::Markov) = push!(f.markov, x)
function _addfact!(f::Facts, x::Facts)
    append!(f.independent, x.independent)
    append!(f.disjoint, x.disjoint)
    append!(f.implies, x.implies)
    append!(f.values, x.values)
    append!(f.iid, x.iid)
    append!(f.markov, x.markov)
end
_addfact!(f::Facts, xs::Union{Tuple,AbstractVector}) = foreach(x -> _addfact!(f, x), xs)
_addfact!(::Facts, x) = throw(ArgumentError(
    "assume takes A ⟂ B (| C), disjoint(…), implies(A, B) and `P(…) => value`; got $(repr(x))"))

_facts(::Nothing) = Facts()
_facts(f::Facts) = f
_facts(x) = assume(x)

Base.isempty(f::Facts) = isempty(f.independent) && isempty(f.disjoint) && isempty(f.implies) &&
                         isempty(f.values) && isempty(f.iid) && isempty(f.markov)
_nfacts(f::Facts) = length(f.independent) + length(f.disjoint) + length(f.implies) +
                    length(f.values) + length(f.iid) + length(f.markov)

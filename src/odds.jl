# Odds, likelihood ratios and Bayes factors: Bayes' rule in odds form,
#
#   P(H | D) / P(Hᶜ | D)  =  P(D | H) / P(D | Hᶜ)  ×  P(H) / P(Hᶜ)
#   posterior odds        =  likelihood ratio     ×  prior odds
#
# and on the log scale, where evidence adds up (the weight of evidence).
# Odds are ordinary quotients of probability terms, so they simplify, compute
# and print like any other expression.

_onetarget(p::Prob) = length(p.targets) == 1 && iseventlike(p.targets[1]) ? p.targets[1] :
    throw(ArgumentError(string("odds are of one event; got ", sprint(show, p))))


"""
    odds(P(H | C)) -> PExpr
    odds(P(H₁ | C), P(H₂ | C)) -> PExpr

The odds of an event, `P(H | C) / P(Hᶜ | C)`, or of one hypothesis against
another, `P(H₁ | C) / P(H₂ | C)`.
"""
function odds(p::Prob)
    h = _onetarget(p)
    return divide(p, _like(p, Atom[complement(h)], p.given))
end
function odds(p::Prob, q::Prob)
    (issetequal(p.given, q.given) && p.name === q.name) ||
        throw(ArgumentError("odds compare two hypotheses under the same conditions"))
    return divide(p, q)
end

"""
    likelihood_ratio(D, H; given = ()) -> PExpr
    likelihood_ratio(D, H₁, H₂; given = ())

`P(D | H) / P(D | Hᶜ)`, how much more probable the evidence `D` is under `H`
than under its complement (or under `H₂`). Also called the Bayes factor.
"""
likelihood_ratio(d, h; given=()) = likelihood_ratio(d, h, complement(h); given)
likelihood_ratio(d, h1, h2; given=()) =
    divide(_mkprob(_aslist(d), vcat(Atom[h1], _aslist(given))), _mkprob(_aslist(d), vcat(Atom[h2], _aslist(given))))


"""
    bayes_factor(D, M₁, M₂; given = ()) -> PExpr

`P(D | M₁) / P(D | M₂)`: the factor by which the data `D` multiply the odds
of model `M₁` against `M₂`.
"""
bayes_factor(d, m1, m2; given=()) = likelihood_ratio(d, m1, m2; given)

"""
    bayes_odds(P(H | D, C)) -> PExpr
    bayes_odds(P(H₁ | D, C), P(H₂ | D, C)) -> PExpr

Bayes' rule in odds form: posterior odds = likelihood ratio × prior odds,
`[P(D | H, C) / P(D | Hᶜ, C)] [P(H | C) / P(Hᶜ | C)]`. The normalising
constant `P(D | C)` cancels, which is why the odds form needs no
denominator. `swap` picks the evidence among the conditioning atoms (by
default all of the observational ones).
"""
function bayes_odds(p::Prob; swap=nothing)
    h = _onetarget(p)
    return bayes_odds(p, _like(p, Atom[complement(h)], p.given); swap)
end
function bayes_odds(p::Prob, q::Prob; swap=nothing)
    issetequal(p.given, q.given) && p.name === q.name ||
        throw(ArgumentError("bayes_odds compares two hypotheses given the same evidence"))
    D = swap === nothing ? Atom[g for g in p.given if !(g isa Intervention)] : _aslist(swap)
    isempty(D) && throw(ArgumentError("nothing to treat as evidence on the right of the bar"))
    C = Atom[g for g in p.given if !(g in D)]
    lr = divide(_like(p, D, vcat(p.targets, C)), _like(p, D, vcat(q.targets, C)))
    prior = divide(_like(p, p.targets, C), _like(p, q.targets, C))
    return mul(lr, prior)
end

"""
    log_odds(P(H | C)) -> PExpr

`log(P(H | C) / P(Hᶜ | C))`, the logit of a probability. With
[`weight_of_evidence`](@ref), log posterior odds = weight of evidence + log
prior odds, so independent pieces of evidence add.
"""
log_odds(p::Prob) = log(odds(p))

"""
    weight_of_evidence(D, H; given = ()) -> PExpr

`log(P(D | H) / P(D | Hᶜ))` (I. J. Good's weight of evidence), the log of the
[`likelihood_ratio`](@ref).
"""
weight_of_evidence(d, h; given=()) = log(likelihood_ratio(d, h; given))

"""
    probability_from_odds(o) -> number
    odds_from_probability(p) -> number

Convert between a probability and odds `o = p / (1 - p)`; exact for
rationals.
"""
probability_from_odds(o::Real) = _div(o, 1 + o)
odds_from_probability(p::Real) = _div(p, 1 - p)

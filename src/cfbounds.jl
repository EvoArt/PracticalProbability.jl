# Probabilities of causation (Tian & Pearl 2000): the probability of
# necessity (PN), of sufficiency (PS) and of necessity and sufficiency (PNS)
# for a binary treatment and outcome. They are not identifiable in general;
# combining experimental and observational data bounds them, and under
# monotonicity (no unit for which the treatment prevents the outcome) the
# bounds meet.

"""
    CFBounds

`max(lower...) ≤ query ≤ min(upper...)`, returned by [`bounds`](@ref).
[`compute`](@ref) gives the numeric interval `(lo, hi)`.
"""
struct CFBounds
    query::PExpr
    lower::Vector{PExpr}
    upper::Vector{PExpr}
end

function Base.show(io::IO, b::CFBounds)
    print(io, "max{", join((sprint(show, e) for e in b.lower), ", "), "} ≤ ", b.query,
          " ≤ min{", join((sprint(show, e) for e in b.upper), ", "), "}")
end
function Base.show(io::IO, ::MIME"text/plain", b::CFBounds)
    println(io, "Bounds on ", b.query, ":")
    println(io, "  ≥ max of  ", join((sprint(show, e) for e in b.lower), ",  "))
    print(io, "  ≤ min of  ", join((sprint(show, e) for e in b.upper), ",  "))
end
_latex(b::CFBounds) = string("\\max\\left\\{", join(map(_latex, b.lower), ", "), "\\right\\} \\le ", _latex(b.query),
                             " \\le \\min\\left\\{", join(map(_latex, b.upper), ", "), "\\right\\}")
latex(b::CFBounds; env::Symbol=:inline) = _wrap(_latex(b), env)

function compute(b::CFBounds, nets::_Nets)
    num(e) = (v = compute(e, nets); v isa Real ? v : throw(ArgumentError("the bounds still depend on $(v.vars)")))
    return (maximum(num, b.lower), minimum(num, b.upper))
end

# The parts of a probability-of-causation query.
function _causation_parts(q::Prob)
    single(a) = a isa Counterfactual && a.hasvalue && length(a.sub) == 1
    if length(q.targets) == 2 && isempty(q.given) && all(single, q.targets)
        a, b = q.targets
        if a.rv == b.rv && a.sub[1].rv == b.sub[1].rv && !isequal(a.sub[1].value, b.sub[1].value) &&
           !isequal(a.value, b.value)
            return (:pns, a.sub[1].rv, a.rv, a.sub[1].value, b.sub[1].value, a.value, b.value)
        end
    elseif length(q.targets) == 1 && single(q.targets[1]) && length(q.given) == 2 && all(a -> a isa Event, q.given)
        t = q.targets[1]
        X, Y = t.sub[1].rv, t.rv
        ex = findfirst(a -> a.rv == X, q.given)
        ey = findfirst(a -> a.rv == Y, q.given)
        if ex !== nothing && ey !== nothing
            xo, yo = q.given[ex].value, q.given[ey].value       # what happened
            xc, yc = t.sub[1].value, t.value                    # the counterfactual
            if !isequal(xo, xc) && !isequal(yo, yc)
                # PN: P(y′_{x′} | x, y) with (x, y) observed; PS: P(y_x | x′, y′).
                # Which one depends only on reading: both have the same shape,
                # (observed x, y; counterfactual x̄, ȳ).
                return (:pn_or_ps, X, Y, xo, xc, yo, yc)
            end
        end
    end
    throw(ArgumentError("bounds takes a probability of causation for a binary treatment X and outcome Y: " *
                        "P(Y(Do(Event(X, x)))(y), Y(Do(Event(X, x′)))(y′)) (PNS) or P(Y(Do(Event(X, x′)))(y′) | X(x), Y(y)) (PN/PS)"))
end

"""
    bounds(query; dag = nothing, monotone = false) -> CFBounds or PExpr

Tian–Pearl bounds on a probability of causation for a binary treatment `X`
(values `x`, `x′`) and outcome `Y` (values `y`, `y′`), from experimental terms
`P(y | do(x))` and observational terms `P(x, y)`:

- PNS, `P(Y(Do(Event(X, x)))(y), Y(Do(Event(X, x′)))(y′))`: the outcome happens with the
  treatment and not without it;
- PN, `P(Y(Do(Event(X, x′)))(y′) | X(x), Y(y))`: for a treated unit with the outcome,
  the outcome would not have happened untreated (probability of necessity);
- PS, `P(Y(Do(Event(X, x)))(y) | X(x′), Y(y′))`: for an untreated unit without the
  outcome, the treatment would have produced it (probability of sufficiency).
  (PN and PS have the same form: observed `(x, y)` and the opposite
  counterfactual; here the first value of each is taken to be the one written
  in the counterfactual's `Do`.)

With `monotone = true` (`Y` is monotone in `X`: no unit with `y_{x′}` and
`y′_x`) the bounds meet and the point value is returned. With `dag` (a DAG or
ADMG) the experimental terms are replaced by their identified observational
expressions. [`compute`](@ref) gives the numeric interval.
"""
function bounds(q::Prob; dag=nothing, monotone::Bool=false)
    parts = _causation_parts(q)
    name = q.name
    # PNS: (x, x′, y, y′). PN/PS: observed (xo, yo) and counterfactual (xc, yc),
    # given as (xo, xc, yo, yc) and written in the PN form P(y′_{x′} | x, y).
    kind, X, Y, x, x′, y, y′ = parts
    pr(atoms...) = P(atoms...; name)
    if kind === :pns
        Pyx, Pyx′, Py′x′ = pr(Y(y) | Do(Event(X, x))), pr(Y(y) | Do(Event(X, x′))), pr(Y(y′) | Do(Event(X, x′)))
        Py = pr(Y(y))
        Pxy, Px′y′, Pxy′, Px′y = pr(X(x), Y(y)), pr(X(x′), Y(y′)), pr(X(x), Y(y′)), pr(X(x′), Y(y))
        if monotone
            out = add(Pyx, -Pyx′)
            return dag === nothing ? out : _identify_terms(out, dag)
        end
        lower = PExpr[Literal(0), add(Pyx, -Pyx′), add(Py, -Pyx′), add(Pyx, -Py)]
        upper = PExpr[Pyx, Py′x′, add(Pxy, Px′y′), add(Pyx, -Pyx′, Pxy′, Px′y)]
    else
        Pyx′ = pr(Y(y) | Do(Event(X, x′)))            # P(y_{x′})
        Py′x′ = pr(Y(y′) | Do(Event(X, x′)))
        Py = pr(Y(y))
        Pxy, Px′y′ = pr(X(x), Y(y)), pr(X(x′), Y(y′))
        if monotone
            out = divide(add(Py, -Pyx′), Pxy)
            return dag === nothing ? out : _identify_terms(out, dag)
        end
        lower = PExpr[Literal(0), divide(add(Py, -Pyx′), Pxy)]
        upper = PExpr[Literal(1), divide(add(Py′x′, -Px′y′), Pxy)]
    end
    b = CFBounds(q, lower, upper)
    dag === nothing && return b
    return CFBounds(q, PExpr[_identify_terms(e, dag) for e in b.lower], PExpr[_identify_terms(e, dag) for e in b.upper])
end

# Replace every interventional term by its identified form.
_asadmg(g::ADMG) = g
_asadmg(g::DAG) = ADMG(g, Tuple{RV,RV}[])
_identify_terms(e::PExpr, g) =
    mapprobs(p -> any(a -> a isa Intervention, p.given) ? identify(p, _asadmg(g)) : p, e)

"""
    pns(X, Y, x, x′, y, y′) -> Prob
    pn(X, Y, x, x′, y, y′) -> Prob
    ps(X, Y, x, x′, y, y′) -> Prob

The probabilities of causation as counterfactual queries: PNS
`P(y_x, y′_{x′})`, PN `P(y′_{x′} | x, y)` and PS `P(y_x | x′, y′)`, for
treatment values `x` (treated), `x′` and outcome values `y` (the outcome
happens), `y′`. Pass them to [`bounds`](@ref), or to [`compute`](@ref) with a
structural model.
"""
pns(X::RV, Y::RV, x, x′, y, y′) = P(Y(Do(Event(X, x)))(y), Y(Do(Event(X, x′)))(y′))
pn(X::RV, Y::RV, x, x′, y, y′) = P(Y(Do(Event(X, x′)))(y′) | (X(x), Y(y)))
ps(X::RV, Y::RV, x, x′, y, y′) = P(Y(Do(Event(X, x)))(y) | (X(x′), Y(y′)))

# ---------------------------------------------------------------------------
# Mediation
# ---------------------------------------------------------------------------

"""
    mediation(Y(y), X, M, x, x′; name = :P) -> (total, direct, indirect)

Pearl's natural effects of changing `X` from `x′` to `x` on the event `Y = y`,
through the mediator `M`, as counterfactual expressions:

- total `P(Y_x = y) - P(Y_{x′} = y)`;
- natural direct effect `P(Y_{x, M_{x′}} = y) - P(Y_{x′} = y)`: change `X`
  but hold `M` at the value it would have had under `x′`;
- natural indirect effect `P(Y_{x′, M_x} = y) - P(Y_{x′} = y)`: keep `X` at
  `x′` but move `M` to its value under `x`.

`identify(e, g)` turns each into observational terms when the graph allows it
(the mediation formula when nothing is confounded), [`compute`](@ref) evaluates
them in a structural model.
"""
function mediation(y::Event, X::RV, M::RV, x, x′; name::Symbol=:P)
    Y, v = y.rv, y.value
    base = P(Y(Do(Event(X, x′)))(v); name)
    total = add(P(Y(Do(Event(X, x)))(v); name), -base)
    direct = add(P(Y(Do(Event(X, x)), M(Do(Event(X, x′))))(v); name), -base)
    indirect = add(P(Y(Do(Event(X, x′)), M(Do(Event(X, x))))(v); name), -base)
    return (total=total, direct=direct, indirect=indirect)
end

"""
    identify(expr, g)

Identify every causal or counterfactual term of an expression (a sum,
product, quotient or marginal of terms) in the graph `g`.
"""
identify(e::Union{Sum,Product,Quotient,Marginal}, g::ADMG) =
    mapprobs(p -> (_hascf(p) || any(a -> a isa Intervention, p.given)) ? identify(p, g) : p, e)
identify(e::Union{Sum,Product,Quotient,Marginal}, g::DAG; latent=RV[]) = identify(e, latent_projection(g, latent))

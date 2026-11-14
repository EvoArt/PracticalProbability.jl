# Probability expressions.
#
# A deliberately small term language rather than a general symbolic one: the
# objects of probability algebra are terms P(targets | given), combined by
# products, quotients, sums and marginalisation (∑ / ∫). Equality is semantic
# where it is cheap to be (the arguments of P are sets, products and sums are
# multisets), which is what lets the rewrite rules in rules.jl match by set
# operations.

"""
    PExpr

Supertype of probability expressions: [`Prob`](@ref), [`Product`](@ref),
[`Quotient`](@ref), [`Sum`](@ref), [`Marginal`](@ref) and [`Literal`](@ref).
"""
abstract type PExpr end

"""
    Prob(targets, given; name = :P)

The term `P(targets | given)`, the probability of all the targets together
(their intersection) given all of `given`. Usually built with [`P`](@ref), or
with a named distribution such as `π(x, y | D)` (see [`Density`](@ref)).

`name` names the distribution. The default `:P` is written `P` for discrete
targets and `p` for continuous ones; any other name (`:π`, `:q`, `:f`) is
written as given. Terms of different distributions never combine.
"""
struct Prob <: PExpr
    targets::Vector{Atom}
    given::Vector{Atom}
    name::Symbol
    function Prob(t::AbstractVector, g::AbstractVector, name::Symbol)
        t = unique(Atom[x for x in t])
        g = unique(Atom[x for x in g])
        isempty(t) && throw(ArgumentError("a probability needs at least one target"))
        any(x -> x isa Intervention, t) &&
            throw(ArgumentError("do(⋅) can only appear on the conditioning side of a probability"))
        return new(t, g, name)
    end
end
Prob(t::AbstractVector, g::AbstractVector; name::Symbol=:P) = Prob(t, g, name)

# A new term of the same distribution as `p`.
_like(p::Prob, t, g) = Prob(t, g, p.name)

"""
    Given(targets, given)

The result of `A | B` on atoms. Only meaningful as an argument to [`P`](@ref)
or [`⟂`](@ref).
"""
struct Given
    targets::Vector{Atom}
    given::Vector{Atom}
end
Base.:|(a::Atom, b::Atom) = Given(Atom[a], Atom[b])
Base.:|(a::Atom, b::Tuple) = Given(Atom[a], _atoms(b))
Base.:|(a::Tuple{Vararg{Atom}}, b::Atom) = Given(_atoms(a), Atom[b])
Base.:|(a::Tuple{Vararg{Atom}}, b::Tuple) = Given(_atoms(a), _atoms(b))
Base.:|(a::AbstractVector{<:Atom}, b::Atom) = Given(_atoms(a), Atom[b])
Base.:|(a::AbstractVector{<:Atom}, b::Union{Tuple,AbstractVector{<:Atom}}) = Given(_atoms(a), _atoms(b))
Base.:|(a::Atom, b::AbstractVector{<:Atom}) = Given(Atom[a], _atoms(b))

# `∪` binds less tightly than `|`, so `P(A | B ∪ C)` arrives as `(A | B) ∪ C`.
# A conditional union an event can only mean the union of the conditioning
# event, so make it that: `(A | B) ∪ C` is `A | (B ∪ C)`.
Base.union(g::Given, bs::Atom...) =
    Given(g.targets, vcat(g.given[1:end-1], Atom[union(g.given[end], bs...)]))

# Intersections inside P are the same as listing the parts: P(A ∩ B) = P(A, B).
function _flatten_inter(xs)
    out = Atom[]
    for x in xs
        if x isa EventIntersection
            append!(out, x.parts)
        else
            push!(out, x)
        end
    end
    return out
end

"""
    _mkprob(targets, given; name) -> PExpr

Build `P(targets | given)` with the event identities applied: intersections
are flattened into the argument lists, `Ω` is dropped, `P(Ω | ⋯) = 1` and
`P(∅ | ⋯) = 0`.
"""
function _mkprob(targets, given; name::Symbol=:P)
    t = Atom[x for x in _flatten_inter(targets) if !(x isa SureEvent)]
    g = Atom[x for x in _flatten_inter(given) if !(x isa SureEvent)]
    any(x -> x isa ImpossibleEvent, g) &&
        throw(ArgumentError("cannot condition on the impossible event ∅"))
    any(x -> x isa ImpossibleEvent, t) && return Literal(0)
    # targets that contradict each other (A and Aᶜ) give probability zero
    _evintersect(Atom[x for x in t if iseventlike(x)]) isa ImpossibleEvent && return Literal(0)
    isempty(t) && return Literal(1)
    return Prob(t, g, name)
end

"""
    P(A, B | C, D)
    P(A ∪ B | C)
    P(Y | Do(X), Z)
    P(X(1) | Z(0))

Build a probability term. Atoms before the bar are the targets (jointly: an
intersection), atoms after it are conditioned on. Write the bar once; anything
after it is conditioned on. Events are named events (`@event A`), `X(value)`,
and their unions, intersections and complements; interventions are `Do(X)`.

Unions and intersections work on both sides of the bar without brackets:
`P(A ∪ B | C)`, `P(A | B ∪ C)` and `P(A | B ∩ C)` mean what they say.
"""
function P(args...; name::Symbol=:P)
    targets = Atom[]
    given = Atom[]
    bar = false
    for a in args
        if a isa Given
            bar && throw(ArgumentError("P(…) can contain only one conditioning bar"))
            append!(targets, a.targets)
            append!(given, a.given)
            bar = true
        elseif a isa Atom
            push!(bar ? given : targets, a)
        elseif a isa Union{Tuple,AbstractVector}
            append!(bar ? given : targets, _atoms(a))
        else
            throw(ArgumentError("P(…) takes random variables, events and do(⋅); got $(repr(a))"))
        end
    end
    return _held() ? _heldprob(targets, given, name) : _mkprob(targets, given; name)
end

# `P(⋯)` inside `@hold`: no identities, except that an intersection already in
# canonical form is listed as separate targets (which prints the same).
function _heldprob(targets, given, name)
    any(x -> x isa ImpossibleEvent, given) &&
        throw(ArgumentError("cannot condition on the impossible event ∅"))
    return Prob(_heldflatten(targets), _heldflatten(given), name)
end
function _heldflatten(xs)
    out = Atom[]
    for x in xs
        x isa EventIntersection && _eventstep_deep(x) === nothing ? append!(out, x.parts) : push!(out, x)
    end
    return out
end

"""
    @hold expr

Evaluate `expr` with the event identities switched off: `'`, `∪`, `∩` and
`P(⋯)` build exactly what is written instead of simplifying on the way, so
`@hold P(A ∩ A')` is the term `P(A ∩ Aᶜ)` rather than `0`. [`simplify`](@ref)
then applies the identities one at a time, and with `trace = true` records each
one (`A ∩ Aᶜ = ∅`, then `P(∅) = 0`):

```julia
e = @hold P(A ∩ A')
Equation(e, simplify(e))          # P(A ∩ Aᶜ) = 0
simplify(e; trace = true)
```

Only construction is held: other operations (`+`, `*`, …) behave as usual, and
`simplify` works with the identities on even when called inside `@hold`.
"""
macro hold(ex)
    return :($task_local_storage(() -> $(esc(ex)), $(QuoteNode(_HOLD_KEY)), true))
end

"""
    Density(name)

A named distribution, called like [`P`](@ref): with `π = Density(:π)`,
`π(x, y | D)` is the joint distribution `π` of `x` and `y` given `D`. Declare
several with [`@density`](@ref). Terms of different distributions are never
combined by the rules (`q(z)` is not `p(z)`).
"""
struct Density
    name::Symbol
end
(d::Density)(args...) = P(args...; name=d.name)
Base.show(io::IO, d::Density) = print(io, d.name, "(⋅)")

"""
    @density π q f

Declare named distributions, binding each name in the calling scope to a
[`Density`](@ref).
"""
macro density(names::Symbol...)
    out = Expr(:block)
    for n in names
        push!(out.args, :($(esc(n)) = $Density($(QuoteNode(n)))))
    end
    push!(out.args, Expr(:tuple, map(esc, names)...))
    return out
end

"""
    @P(A, B | C)

Like [`P`](@ref), but bare names that are not already bound to an [`RV`](@ref)
become discrete random variables of that name, so nothing has to be declared.
"""
macro P(args...)
    return :($P($(map(_pmacro_walk, args)...)))
end

_pmacro_walk(x) = esc(x)
_pmacro_walk(s::Symbol) = _rvref(s)
function _pmacro_walk(ex::Expr)
    if ex.head === :call
        f = ex.args[1]
        if f in (:|, :Do, :⟂, :∪, :∩, :!, :setdiff)
            return Expr(:call, esc(f), map(_pmacro_walk, ex.args[2:end])...)
        elseif f isa Symbol            # `X(1)`: an event; the value is ordinary code
            return Expr(:call, _rvref(f), map(esc, ex.args[2:end])...)
        end
        return esc(ex)
    elseif ex.head === :tuple
        return Expr(:tuple, map(_pmacro_walk, ex.args)...)
    elseif ex.head === Symbol("'")
        return Expr(Symbol("'"), _pmacro_walk(ex.args[1]))
    end
    return esc(ex)
end

"""
    Literal(value)

A numeric constant inside an expression.
"""
struct Literal <: PExpr
    value::Number
end

"""
    Product(factors)

A product of factors. Build with `*`, which flattens nested products and folds
constants.
"""
struct Product <: PExpr
    factors::Vector{PExpr}
end

"""
    Quotient(num, den)

`num / den`. Build with `/`.
"""
struct Quotient <: PExpr
    num::PExpr
    den::PExpr
end

"""
    Sum(terms)

A sum of terms. Build with `+` and `-`, which flatten nested sums and collect
like terms (`P(A) + P(A) = 2 P(A)`).
"""
struct Sum <: PExpr
    terms::Vector{PExpr}
end

"""
    Marginal(over, body)

`body` summed (discrete variables) or integrated (continuous variables) over
the variables `over`. Build with [`Σ`](@ref) or [`marginalize`](@ref).
"""
struct Marginal <: PExpr
    over::Vector{RV}
    body::PExpr
end

# ---------------------------------------------------------------------------
# Smart constructors (normalising: flatten, drop ones, fold constants)
# ---------------------------------------------------------------------------

mul() = Literal(1)
function mul(xs::PExpr...)
    fs = PExpr[]
    c = 1
    for x in xs
        for f in (x isa Product ? x.factors : (x,))
            if f isa Literal
                c *= f.value
            else
                push!(fs, f)
            end
        end
    end
    c == 0 && return Literal(0)
    c != 1 && pushfirst!(fs, Literal(c))
    isempty(fs) && return Literal(c)
    length(fs) == 1 && return fs[1]
    return Product(fs)
end

# A term as coefficient × the rest.
function _coef(t::PExpr)
    t isa Literal && return (t.value, Literal(1))
    if t isa Product && t.factors[1] isa Literal
        return (t.factors[1].value, mul(t.factors[2:end]...))
    end
    return (1, t)
end

function add(xs::PExpr...)
    rests = PExpr[]
    coefs = Number[]
    for x in xs
        for t in (x isa Sum ? x.terms : (x,))
            c, r = _coef(t)
            i = findfirst(==(r), rests)
            if i === nothing
                push!(rests, r)
                push!(coefs, c)
            else
                coefs[i] += c
            end
        end
    end
    terms = PExpr[]
    k = findfirst(==(Literal(1)), rests)       # the constant goes first
    k !== nothing && coefs[k] != 0 && push!(terms, Literal(coefs[k]))
    for (c, r) in zip(coefs, rests)
        (r == Literal(1) || c == 0) && continue
        push!(terms, mul(Literal(c), r))
    end
    isempty(terms) && return Literal(0)
    length(terms) == 1 && return terms[1]
    return Sum(terms)
end

function divide(n::PExpr, d::PExpr)
    d isa Literal && d.value == 1 && return n
    n isa Literal && n.value == 0 && return n
    n isa Literal && d isa Literal && return Literal(n.value / d.value)
    return Quotient(n, d)
end

function marg(over, body::PExpr)
    vs = unique(RV[v for v in over])
    isempty(vs) && return body
    body isa Literal && body.value == 0 && return body
    if body isa Marginal
        return Marginal(unique(vcat(body.over, vs)), body.body)
    end
    return Marginal(vs, body)
end

_lit(x::Number) = Literal(x)
Base.:*(a::PExpr, b::PExpr) = mul(a, b)
Base.:*(a::Number, b::PExpr) = mul(Literal(a), b)
Base.:*(a::PExpr, b::Number) = mul(a, Literal(b))
Base.:/(a::PExpr, b::PExpr) = divide(a, b)
Base.:/(a::Number, b::PExpr) = divide(Literal(a), b)
Base.:/(a::PExpr, b::Number) = divide(a, Literal(b))
Base.:+(a::PExpr, b::PExpr) = add(a, b)
Base.:+(a::Number, b::PExpr) = add(Literal(a), b)
Base.:+(a::PExpr, b::Number) = add(a, Literal(b))
Base.:-(a::PExpr) = mul(Literal(-1), a)
Base.:-(a::PExpr, b::PExpr) = add(a, -b)
Base.:-(a::Number, b::PExpr) = add(Literal(a), -b)
Base.:-(a::PExpr, b::Number) = add(a, Literal(-b))
Base.prod(xs::AbstractVector{<:PExpr}) = mul(xs...)
Base.sum(xs::AbstractVector{<:PExpr}) = add(xs...)

"""
    Σ(vars, expr)
    Σ(A, expr)

Marginalise `expr` over `vars`: a sum for discrete variables and an integral
for continuous ones.
"""
Σ(v::RV, e::PExpr) = marg(RV[v], e)
Σ(vs::Union{Tuple,AbstractVector}, e::PExpr) = marg(RV[v for v in vs], e)

# ---------------------------------------------------------------------------
# Equality and hashing (P arguments are sets; products and sums are multisets)
# ---------------------------------------------------------------------------

Base.:(==)(a::Prob, b::Prob) =
    a.name === b.name && issetequal(a.targets, b.targets) && issetequal(a.given, b.given)
Base.hash(a::Prob, h::UInt) = hash(a.name, hash(Set(a.given), hash(Set(a.targets), hash(:Prob, h))))

Base.:(==)(a::Literal, b::Literal) = a.value == b.value
Base.hash(a::Literal, h::UInt) = hash(a.value, hash(:Literal, h))

Base.:(==)(a::Quotient, b::Quotient) = a.num == b.num && a.den == b.den
Base.hash(a::Quotient, h::UInt) = hash(a.den, hash(a.num, hash(:Quotient, h)))

Base.:(==)(a::Marginal, b::Marginal) = issetequal(a.over, b.over) && a.body == b.body
Base.hash(a::Marginal, h::UInt) = hash(a.body, hash(Set(a.over), hash(:Marginal, h)))

function _multiset_eq(xs, ys)
    length(xs) == length(ys) || return false
    rest = copy(ys)
    for f in xs
        i = findfirst(==(f), rest)
        i === nothing && return false
        deleteat!(rest, i)
    end
    return true
end
_multiset_hash(xs) = (acc = zero(UInt); for f in xs; acc += hash(f); end; acc)

Base.:(==)(a::Product, b::Product) = _multiset_eq(a.factors, b.factors)
Base.hash(a::Product, h::UInt) = hash(_multiset_hash(a.factors), hash(:Product, h))
Base.:(==)(a::Sum, b::Sum) = _multiset_eq(a.terms, b.terms)
Base.hash(a::Sum, h::UInt) = hash(_multiset_hash(a.terms), hash(:Sum, h))

Base.:(==)(::PExpr, ::PExpr) = false

# ---------------------------------------------------------------------------
# Variables of an expression
# ---------------------------------------------------------------------------

"""
    freevars(expr) -> Vector{RV}

The random variables an expression is a function of: those that appear as
plain variables (not fixed by an event) and are not summed or integrated out.
"""
freevars(e::PExpr) = sort!(collect(_freevars(e)))
_freevars(::Literal) = Set{RV}()
function _freevars(p::Prob)
    s = Set{RV}()
    for a in Iterators.flatten((p.targets, p.given))
        v = freerv(a)
        v === nothing || push!(s, v)
    end
    return s
end
_freevars(p::Product) = mapreduce(_freevars, union, p.factors; init=Set{RV}())
_freevars(p::Sum) = mapreduce(_freevars, union, p.terms; init=Set{RV}())
_freevars(q::Quotient) = union(_freevars(q.num), _freevars(q.den))
_freevars(m::Marginal) = setdiff(_freevars(m.body), m.over)

# Every variable mentioned, fixed by an event or not (bound variables included).
allvars(e::PExpr) = sort!(collect(_allvars(e)))
_allvars(::Literal) = Set{RV}()
_allvars(p::Prob) = Set{RV}(v for a in Iterators.flatten((p.targets, p.given)) for v in rvs(a))
_allvars(p::Product) = mapreduce(_allvars, union, p.factors; init=Set{RV}())
_allvars(p::Sum) = mapreduce(_allvars, union, p.terms; init=Set{RV}())
_allvars(q::Quotient) = union(_allvars(q.num), _allvars(q.den))
_allvars(m::Marginal) = union(_allvars(m.body), Set(m.over))

mentions(e::PExpr, v::RV) = v in _allvars(e)

# Children, for generic traversal.
children(::Union{Prob,Literal}) = PExpr[]
children(p::Product) = p.factors
children(s::Sum) = s.terms
children(q::Quotient) = PExpr[q.num, q.den]
children(m::Marginal) = PExpr[m.body]

rebuild(p::Product, cs) = mul(cs...)
rebuild(s::Sum, cs) = add(cs...)
rebuild(q::Quotient, cs) = divide(cs[1], cs[2])
rebuild(m::Marginal, cs) = marg(m.over, cs[1])

"""
    mapprobs(f, expr)

Apply `f` to every [`Prob`](@ref) term in `expr` and rebuild.
"""
mapprobs(f, p::Prob) = f(p)
mapprobs(f, l::Literal) = l
mapprobs(f, e::PExpr) = rebuild(e, PExpr[mapprobs(f, c) for c in children(e)])

"""
    substitute(expr, rules::Pair...)

Replace atoms throughout `expr`: `substitute(e, X => X(1))` fixes `X` to `1`
wherever it appears free, `substitute(e, A => A')` swaps an event for its
complement, also inside compound events. Variables bound by a ∑/∫ are not
replaced inside it.
"""
substitute(e::PExpr, rules::Pair...) = _subs(e, Dict{Atom,Atom}(rules...))

_sa(a::Union{RV,Event,SureEvent,ImpossibleEvent}, d) = get(d, a, a)
_sa(a::Intervention, d) = (t = _sa(a.target, d); t isa Intervention ? t : Intervention(t))
_sa(a::Complement, d) = haskey(d, a) ? d[a] : _complement(_sa(a.event, d))
_sa(a::EventUnion, d) = haskey(d, a) ? d[a] : _evunion(Atom[_sa(x, d) for x in a.parts])
_sa(a::EventIntersection, d) = haskey(d, a) ? d[a] : _evintersect(Atom[_sa(x, d) for x in a.parts])

_subs(l::Literal, d) = l
_subs(p::Prob, d) = _mkprob(Atom[_sa(a, d) for a in p.targets], Atom[_sa(a, d) for a in p.given]; name=p.name)
_subs(e::Union{Product,Sum,Quotient}, d) = rebuild(e, PExpr[_subs(c, d) for c in children(e)])
function _subs(m::Marginal, d)
    d2 = filter(kv -> !(kv.first in m.over), d)
    return marg(m.over, _subs(m.body, d2))
end

# ---------------------------------------------------------------------------
# Equations
# ---------------------------------------------------------------------------

"""
    Equation(lhs, rhs; relation = :eq)

`lhs = rhs` (or `lhs ∝ rhs` with `relation = :propto`). Returned by
[`posterior`](@ref) and by `Equation(::Derivation)`.
"""
struct Equation
    lhs::PExpr
    rhs::PExpr
    relation::Symbol
end
Equation(lhs::PExpr, rhs::PExpr; relation::Symbol=:eq) = Equation(lhs, rhs, relation)

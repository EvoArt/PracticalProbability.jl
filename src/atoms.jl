# Atoms: the things that can appear inside P( … | … ).
#
#   RV            a random variable `X` (free: ranges over its values), or a
#                 named event `A` (kind :event: "A happens")
#   Event         a variable fixed to a value, `X(1)` (displayed `X=1`)
#   compound      events built with ∪, ∩ and complement: `A ∪ B`, `Aᶜ` (`A'`)
#   Ω, ∅          the sure and the impossible event
#   Intervention  `Do(X)` / `Do(X(1))`, only allowed to the right of the bar

"""
    Atom

Supertype of everything that can appear inside a probability term: [`RV`](@ref),
[`Event`](@ref), compound events (`A ∪ B`, `A ∩ B`, `Aᶜ`), `Ω`, `∅` and
[`Intervention`](@ref).
"""
abstract type Atom end

"""
    RV(name::Symbol, kind::Symbol = :discrete)

A random variable, or a named event. `kind` is

- `:discrete`: summed over (`∑`), written with a capital `P`;
- `:continuous`: integrated over (`∫ … dθ`), written with a density `p`;
- `:event`: a named event (a yes/no variable). `P(A)` is the probability that
  `A` happens. Events combine with `∪`, `∩` and complement (`A'`, `!A`, or
  [`complement`](@ref)).

Calling a variable with a value gives an [`Event`](@ref): `X(1)` is `X = 1`.

Declare several at once with [`@rv`](@ref) or [`@event`](@ref).
"""
struct RV <: Atom
    name::Symbol
    kind::Symbol
    function RV(name::Symbol, kind::Symbol=:discrete)
        kind in (:discrete, :continuous, :event) ||
            throw(ArgumentError("RV kind must be :discrete, :continuous or :event, got :$kind"))
        return new(name, kind)
    end
end
RV(name::AbstractString, kind::Symbol=:discrete) = RV(Symbol(name), kind)

"""
    Event(rv, value)

The event `rv = value`. Usually written `X(value)`.
"""
struct Event{T} <: Atom
    rv::RV
    value::T
end
(x::RV)(v) = Event(x, v)

Base.:(==)(a::Event, b::Event) = a.rv == b.rv && isequal(a.value, b.value)
Base.hash(a::Event, h::UInt) = hash(a.value, hash(a.rv, hash(:Event, h)))

"""
    Intervention(target)
    Do(X)

The intervention `do(X)` (Pearl). Only valid on the conditioning side of a
probability: `P(Y | Do(X))`. `do` is a Julia keyword, hence the capital `D`.
`Do(X, Z)` returns a tuple of interventions.
"""
struct Intervention <: Atom
    target::Union{RV,Event}
end
Do(x::Union{RV,Event}) = Intervention(x)
Do(xs::Union{RV,Event}...) = map(Intervention, xs)

isdiscrete(x::RV) = x.kind === :discrete || x.kind === :event
iscontinuous(x::RV) = x.kind === :continuous
isevent(x::RV) = x.kind === :event

Base.isless(a::RV, b::RV) = isless(a.name, b.name)
Base.nameof(x::RV) = x.name

# ---------------------------------------------------------------------------
# Compound events
# ---------------------------------------------------------------------------

"""
    SureEvent  (`Ω`)
    ImpossibleEvent  (`∅`)

The whole sample space and the empty event. `P(Ω | C) = 1`, `P(∅ | C) = 0`.
"""
struct SureEvent <: Atom end
struct ImpossibleEvent <: Atom end
const Ω = SureEvent()
const ∅ = ImpossibleEvent()

"""
    EventUnion(parts)       `A ∪ B`
    EventIntersection(parts) `A ∩ B`
    Complement(event)        `Aᶜ`, typed `A'` or `!A`

Compound events. Build them with `∪`, `∩`, `'`/`!` and `setdiff` (`A ∖ B` is
`A ∩ Bᶜ`); the constructors apply the trivial identities on the way
(`(Aᶜ)ᶜ = A`, `A ∪ A = A`, `A ∩ Aᶜ = ∅`, `A ∪ Ω = Ω`, …). De Morgan is not
automatic: see [`demorgan`](@ref). Inside [`@hold`](@ref) they build exactly
what was written, and `simplify` applies the identities step by step.
"""
struct EventUnion <: Atom
    parts::Vector{Atom}
end
struct EventIntersection <: Atom
    parts::Vector{Atom}
end
struct Complement <: Atom
    event::Atom
end

for T in (:EventUnion, :EventIntersection)
    @eval Base.:(==)(a::$T, b::$T) = issetequal(a.parts, b.parts)
    @eval Base.hash(a::$T, h::UInt) = hash(Set(a.parts), hash($(QuoteNode(T)), h))
end
Base.:(==)(a::Complement, b::Complement) = a.event == b.event
Base.hash(a::Complement, h::UInt) = hash(a.event, hash(:Complement, h))

const CompoundEvent = Union{EventUnion,EventIntersection,Complement,SureEvent,ImpossibleEvent}

"""
    iseventlike(atom) -> Bool

Whether an atom denotes an event (something that happens or not): a named
event, `X = x`, a compound event, `Ω` or `∅`. A plain discrete or continuous
variable is not an event; it ranges over its values.
"""
iseventlike(x::RV) = isevent(x)
iseventlike(::Event) = true
iseventlike(::CompoundEvent) = true
iseventlike(::Atom) = false

function _check_event(a)
    a isa Atom && iseventlike(a) && return a
    throw(ArgumentError("$(a) is not an event: ∪, ∩ and complements apply to named events " *
                        "(@event A), to `X(value)` and to their combinations"))
end

"""
    complement(E)

The complement `Eᶜ`, also written `E'` or `!E`.
"""
complement(e::Atom) = _held() ? Complement(_check_event(e)) : _complement(_check_event(e))
_complement(c::Complement) = c.event
_complement(::SureEvent) = ∅
_complement(::ImpossibleEvent) = Ω
_complement(e::Atom) = Complement(e)
Base.adjoint(e::Atom) = complement(e)
Base.:!(e::Atom) = complement(e)

_iscomplementpair(a, b) = (a isa Complement && a.event == b) || (b isa Complement && b.event == a)

# Held construction (`@hold`, expr.jl): `'`, `∪`, `∩` and `P` build exactly what
# was written, and `simplify` applies the identities one step at a time
# (`rule_events` in rules.jl). Task-local, so other tasks are unaffected.
# Internal code calls `_complement`, `_evunion`, `_evintersect` and `_mkprob`
# directly and always gets canonical forms.
const _HOLD_KEY = :PracticalProbability_hold
_held() = (s = current_task().storage; s !== nothing && get(s, _HOLD_KEY, false)::Bool)
_unheld(f) = _held() ? task_local_storage(f, _HOLD_KEY, false) : f()

# A compound of `parts` with no identities applied except associativity, which
# is notation (`A ∩ B ∩ C` parses as `(A ∩ B) ∩ C`).
function _rawcompound(::Type{T}, parts) where {T}
    length(parts) == 1 && return parts[1]
    return T(reduce(vcat, (p isa T ? p.parts : Atom[p] for p in parts); init=Atom[]))
end

function _evunion(parts)
    ps = Atom[]
    for p in parts
        for q in (p isa EventUnion ? p.parts : (p,))
            q isa ImpossibleEvent && continue
            q isa SureEvent && return Ω
            q in ps || push!(ps, q)
        end
    end
    any(i -> any(j -> _iscomplementpair(ps[i], ps[j]), i+1:length(ps)), eachindex(ps)) && return Ω
    isempty(ps) && return ∅
    length(ps) == 1 && return ps[1]
    return EventUnion(ps)
end

function _evintersect(parts)
    ps = Atom[]
    for p in parts
        for q in (p isa EventIntersection ? p.parts : (p,))
            q isa SureEvent && continue
            q isa ImpossibleEvent && return ∅
            q in ps || push!(ps, q)
        end
    end
    any(i -> any(j -> _iscomplementpair(ps[i], ps[j]), i+1:length(ps)), eachindex(ps)) && return ∅
    isempty(ps) && return Ω
    length(ps) == 1 && return ps[1]
    return EventIntersection(ps)
end

Base.union(a::Atom, bs::Atom...) =
    (ps = map(_check_event, (a, bs...)); _held() ? _rawcompound(EventUnion, ps) : _evunion(ps))
Base.intersect(a::Atom, bs::Atom...) =
    (ps = map(_check_event, (a, bs...)); _held() ? _rawcompound(EventIntersection, ps) : _evintersect(ps))
Base.setdiff(a::Atom, b::Atom) = intersect(a, complement(b))

# ---------------------------------------------------------------------------
# Which variables an atom is about
# ---------------------------------------------------------------------------

"""
    rv(atom) -> RV

The random variable an atom is about (for compound events, see `rvs`).
"""
rv(x::RV) = x
rv(e::Event) = e.rv
rv(i::Intervention) = rv(i.target)
function rv(a::Atom)
    vs = rvs(a)
    length(vs) == 1 && return vs[1]
    throw(ArgumentError("$(a) involves $(length(vs)) variables; use rvs"))
end

"""
    rvs(atom) -> Vector{RV}

Every random variable an atom involves.
"""
rvs(x::Union{RV,Event,Intervention}) = RV[rv(x)]
rvs(c::Complement) = rvs(c.event)
rvs(u::Union{EventUnion,EventIntersection}) = unique(reduce(vcat, map(rvs, u.parts); init=RV[]))
rvs(::Union{SureEvent,ImpossibleEvent}) = RV[]

# Free (unbound, unfixed) random variables of an atom: events fix theirs.
freerv(x::RV) = isevent(x) ? nothing : x
freerv(::Event) = nothing
freerv(i::Intervention) = freerv(i.target)
freerv(::Atom) = nothing

_atoms(x::Atom) = Atom[x]
_atoms(x::Union{Tuple,AbstractVector}) = Atom[a for a in x]

# ---------------------------------------------------------------------------
# Declaration macros
# ---------------------------------------------------------------------------

function _declare(args, default::Symbol, macroname)
    out = Expr(:block)
    names = Symbol[]
    for a in args
        if a isa Symbol
            push!(out.args, :($(esc(a)) = $RV($(QuoteNode(a)), $(QuoteNode(default)))))
            push!(names, a)
        elseif a isa Expr && a.head === :(::) && length(a.args) == 2 && a.args[1] isa Symbol
            kind = a.args[2]
            kind in (:discrete, :continuous, :event) ||
                error("$macroname: kind must be `discrete`, `continuous` or `event`, got `$kind`")
            push!(out.args, :($(esc(a.args[1])) = $RV($(QuoteNode(a.args[1])), $(QuoteNode(kind)))))
            push!(names, a.args[1])
        else
            error("$macroname: expected `name` or `name::kind`, got `$a`")
        end
    end
    push!(out.args, Expr(:tuple, map(esc, names)...))
    return out
end

"""
    @rv X Y θ::continuous A::event

Declare random variables, binding each name in the calling scope. Variables are
discrete unless annotated `::continuous` or `::event`. Returns them as a tuple.
"""
macro rv(args...)
    return _declare(args, :discrete, "@rv")
end

"""
    @event A B C

Declare named events (random variables of kind `:event`), binding each name in
the calling scope. Same as `@rv A::event B::event C::event`.
"""
macro event(args...)
    return _declare(args, :event, "@event")
end

# Inside macros (`@P`, `@dag`, …) a bare symbol means "the RV bound to this name
# if there is one, otherwise a fresh discrete RV of that name".
function _rvref(s::Symbol)
    return esc(:((@isdefined($s) && $s isa $RV) ? $s : $RV($(QuoteNode(s)))))
end

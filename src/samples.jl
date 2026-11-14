# Events defined by conditions on draws (posterior samples, simulations): "σ > 0.2",
# "μ is positive". Each becomes a named event with a yes/no value per draw, so
# probabilities are frequencies over the draws, and `venn`, `probtree` and
# `mosaic` can draw them.

"""
    SampleEvents

Named events, each defined by a yes/no value per draw: the `events`, their
`masks` (one `BitVector` per event, all the same length), and `conditions` (the
condition as written, e.g. `:(σ > 0.2)`, or `nothing`). Build with
[`@sampleevents`](@ref) or [`sample_events`](@ref).

A probability of these events is a frequency over the draws:
`compute(P(A | B), se)` is the share of draws where `B` holds in which `A`
also holds. `cell_probabilities(se)` gives every cell of their Venn diagram, and
[`venn`](@ref), [`probtree`](@ref) and [`mosaic`](@ref) take `samples = se`.
"""
struct SampleEvents
    events::Vector{RV}
    masks::Vector{BitVector}
    conditions::Vector{Any}
    function SampleEvents(events, masks, conditions)
        length(events) == length(masks) == length(conditions) ||
            throw(ArgumentError("one mask and one condition per event"))
        all(e -> e isa RV && isevent(e), events) ||
            throw(ArgumentError("sample events must be named events (@event A)"))
        allunique(events) || throw(ArgumentError("each event can be defined once"))
        isempty(masks) || all(m -> length(m) == length(masks[1]), masks) ||
            throw(ArgumentError("every event needs a value for every draw: masks have different lengths"))
        return new(RV[e for e in events], BitVector[BitVector(vec(m)) for m in masks], Any[c for c in conditions])
    end
end

"""
    ndraws(se::SampleEvents) -> Int

The number of draws.
"""
ndraws(se::SampleEvents) = isempty(se.masks) ? 0 : length(se.masks[1])

"""
    sample_events(A => mask, B => mask; conditions = nothing) -> SampleEvents

[`SampleEvents`](@ref) from yes/no values per draw: `mask` is a vector of
`Bool`s, one per draw, such as `draws.σ .> 0.2`. `A` must be a named event
(`@event A`). `conditions` (a vector, one per event) describes each event for
labels. [`@sampleevents`](@ref) does this from the conditions themselves.
"""
function sample_events(pairs::Pair...; conditions=nothing)
    evs = RV[p.first for p in pairs]
    masks = [BitVector(vec(collect(Bool, p.second))) for p in pairs]
    cond = conditions === nothing ? fill(nothing, length(pairs)) : collect(conditions)
    return SampleEvents(evs, masks, cond)
end

"""
    @sampleevents draws begin
        A = σ > 0.2
        B = μ > 0
    end

Define named events by conditions on draws. Each line binds a new named event
(`A`, `B`) in the calling scope and evaluates its condition draw by draw. Names
in a condition are columns of `draws` (a Tables.jl table such as a DataFrame or
an MCMCChains chain, a `NamedTuple` or a `Dict` of vectors), or ordinary
variables if no column has that name. Conditions are broadcast (`abs(μ) < 1`,
`0 < μ < 1`, `(μ > 0) & (σ < 1)`; `&&`/`||` work too), and `θ[1]` reads the
column named `θ[1]`. Returns the [`SampleEvents`](@ref).

```julia
@sampleevents chain begin
    A = σ > 0.2
    B = μ > 0
end
venn(P(A | B); samples = ans)
```
"""
macro sampleevents(draws, block)
    stmts = block isa Expr && block.head === :block ? filter(x -> !(x isa LineNumberNode), block.args) : Any[block]
    d = gensym(:draws)
    out = Expr(:block, :(local $d = $draws))
    evs, masks, conds = Symbol[], Symbol[], Any[]
    for s in stmts
        s isa Expr && s.head === :(=) && s.args[1] isa Symbol ||
            error("@sampleevents: expected `Name = condition`, got `$(s)`")
        name, cond = s.args[1], s.args[2]
        cols = Dict{Symbol,Symbol}()
        body = _sample_rewrite(cond, cols)
        for (c, g) in cols
            push!(out.args, :(local $g = $(_samplecol)($d, $(QuoteNode(c)), () -> $c)))
        end
        m = gensym(name)
        push!(out.args, :(local $m = $(_samplemask)(Base.@__dot__($body), $(QuoteNode(name)))))
        push!(out.args, :($name = $(RV)($(QuoteNode(name)), :event)))
        push!(evs, name)
        push!(masks, m)
        push!(conds, cond)
    end
    push!(out.args, :($(SampleEvents)($(RV)[$(evs...)], [$(masks...)], Any[$(map(QuoteNode, conds)...)])))
    return esc(out)
end

# Replace the column names in a condition by local variables (recorded in
# `cols`), leaving function names, operators and literals alone. `&&`/`||`
# become `&`/`|` so the condition can be broadcast.
function _sample_rewrite(ex, cols)
    if ex isa Symbol
        ex in (:true, :false, :nothing, :missing) && return ex
        return get!(() -> gensym(ex), cols, ex)
    elseif ex isa Expr
        if ex.head === :call
            return Expr(:call, ex.args[1], (_sample_rewrite(a, cols) for a in ex.args[2:end])...)
        elseif ex.head === :comparison       # 0 < μ < 1: operators sit at even positions
            return Expr(:comparison, (iseven(k) ? a : _sample_rewrite(a, cols) for (k, a) in enumerate(ex.args))...)
        elseif ex.head in (:&&, :||)
            return Expr(:call, ex.head === :&& ? :& : :|, (_sample_rewrite(a, cols) for a in ex.args)...)
        elseif ex.head === :ref && ex.args[1] isa Symbol && all(a -> a isa Union{Integer,Symbol}, ex.args[2:end])
            return get!(() -> gensym(ex.args[1]), cols, Symbol(string(ex)))
        end
        return Expr(ex.head, (_sample_rewrite(a, cols) for a in ex.args)...)
    end
    return ex
end

function _samplecol(d, s::Symbol, fallback)
    col = _getcolumn(d, s)
    return col === nothing ? fallback() : col
end

_getcolumn(d::AbstractDict, s::Symbol) =
    haskey(d, s) ? vec(collect(d[s])) : haskey(d, string(s)) ? vec(collect(d[string(s)])) : nothing
_getcolumn(d::NamedTuple, s::Symbol) = haskey(d, s) ? vec(collect(d[s])) : nothing
function _getcolumn(d, s::Symbol)
    if Tables.istable(d)
        cols = Tables.columns(d)
        return s in Tables.columnnames(cols) ? vec(collect(Tables.getcolumn(cols, s))) : nothing
    end
    try
        return vec(collect(d[s]))           # e.g. chains indexed by parameter name
    catch
        return nothing
    end
end

function _samplemask(m, name)
    m isa AbstractArray ||
        throw(ArgumentError("the condition for $(name) gives one value, not one per draw: " *
                            "does it mention a column of the draws?"))
    return BitVector(vec(m))
end

# ---------------------------------------------------------------------------
# Probabilities as frequencies
# ---------------------------------------------------------------------------

function _drawmask(a::Atom, se::SampleEvents)
    a isa SureEvent && return trues(ndraws(se))
    a isa ImpossibleEvent && return falses(ndraws(se))
    k = findfirst(==(a), se.events)
    k !== nothing && return se.masks[k]
    a isa Complement && return .!_drawmask(a.event, se)
    a isa EventUnion && return reduce(.|, (_drawmask(x, se) for x in a.parts))
    a isa EventIntersection && return reduce(.&, (_drawmask(x, se) for x in a.parts))
    throw(ArgumentError("$(a) is not defined by the draws (the sample events are " *
                        "$(join((e.name for e in se.events), ", ")))"))
end
_drawmask(atoms::AbstractVector, se) = reduce(.&, (_drawmask(a, se) for a in atoms); init=trues(ndraws(se)))

_seval(l::Literal, se) = Float64(l.value)
function _seval(p::Prob, se)
    g = _drawmask(p.given, se)
    n = count(g)
    n == 0 && throw(ArgumentError("no draw satisfies the condition of $(p)"))
    return count(_drawmask(p.targets, se) .& g) / n
end
_seval(s::Sum, se) = sum(_seval(t, se) for t in s.terms)
_seval(p::Product, se) = prod(_seval(f, se) for f in p.factors)
_seval(q::Quotient, se) = _seval(q.num, se) / _seval(q.den, se)
_seval(e::PExpr, se) = throw(ArgumentError("$(e) cannot be computed from draws of events"))

"""
    compute(expr, se::SampleEvents) -> Float64

The value of an expression about sample events, with every probability a
frequency over the draws: `P(A | B)` is the share of the draws where `B` holds
in which `A` holds too. Sums, products and quotients of such terms work too.
"""
compute(e::PExpr, se::SampleEvents) = _seval(e, se)
compute(eq::Equation, se::SampleEvents) = (compute(eq.lhs, se), compute(eq.rhs, se))

"""
    cell_probabilities(se::SampleEvents; events = se.events) -> (events, probs)

The share of draws in every cell of the Venn diagram of the events, in the same
form as [`cell_probabilities`](@ref) for facts: cell `k` is where `events[i]`
holds iff bit `i - 1` of `k - 1` is set.
"""
function cell_probabilities(se::SampleEvents; events=nothing)
    evs = events === nothing ? se.events : RV[e for e in events]
    masks = [_drawmask(e, se) for e in evs]
    n = ndraws(se)
    counts = zeros(Int, 1 << length(evs))
    for d in 1:n
        k = 0
        for (i, m) in enumerate(masks)
            m[d] && (k |= 1 << (i - 1))
        end
        counts[k+1] += 1
    end
    return (evs, Union{Nothing,Float64}[c / n for c in counts])
end

_conditionstring(c) = c === nothing ? "" : string(c)
# LaTeX for a condition, for labels: `σ > 0.2` → `\sigma > 0.2`.
_conditionlatex(c) = c === nothing ? nothing : (try
    String(latexify(c; env=:raw))
catch
    nothing
end)

function Base.show(io::IO, se::SampleEvents)
    print(io, "SampleEvents(", ndraws(se), " draws")
    for (e, c, m) in zip(se.events, se.conditions, se.masks)
        print(io, "; ", e.name, c === nothing ? "" : " = $(c)", ": ", _num3(count(m) / max(1, length(m))))
    end
    print(io, ")")
end

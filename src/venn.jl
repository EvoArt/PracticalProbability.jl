# Venn diagrams of event probabilities.
#
# A Venn diagram shows events, so only expressions about events have one:
# named events (`@event A`), `X(value)`, and their unions, intersections and
# complements. Each probability term `P(E | C)` becomes one panel: the cells of
# `E ∩ C` are highlighted, `C` is the sample space that remains, and whatever
# lies outside `C` is greyed out. Sums, products, quotients, equations and
# derivations are laid out as rows of panels joined by their operators, so
# `P(A ∪ B) = P(A) + P(B) - P(A ∩ B)` is drawn as a picture of itself.
#
# This file is the pure-Julia part: which circles to draw, which cells to
# highlight, and where everything goes. The drawing is in src/draw/venn.jl
# (PracticalBayesPlot) and in the Makie extension.

const VENN_MAXSETS = 3

# ---------------------------------------------------------------------------
# Plot functions (drawn by src/draw/venn.jl or the Makie extension)
# ---------------------------------------------------------------------------

"""
    venn(x; sets = nothing, scale = 200, figure = (;), axis = (;), kwargs...)

Draw `x` as Venn diagrams and return the figure: a `PracticalBayesPlot.Figure`,
or with the Makie backend a `Makie.FigureAxisPlot` (see [`plotbackend`](@ref)).
`x` is a probability expression about events, an [`Equation`](@ref), a
[`Derivation`](@ref) or a [`VennLayout`](@ref).

Each probability `P(E | C)` is one panel. The cells of `E ∩ C` are filled in,
the rest of `C` is lightly tinted and everything outside `C` is greyed out, so
conditioning visibly shrinks the sample space. Mutually exclusive events of the
same variable (`X(1)` and `X(2)`) leave their impossible overlap hatched. Sums,
products, quotients and equations are drawn with their operators between the
panels; a derivation is drawn one line per step.

Only events have a Venn diagram, and at most $(VENN_MAXSETS) of them per
figure. Plain random variables (`P(X)`, which ranges over the values of `X`),
`do(⋅)` and sums over variables raise an `ArgumentError`.

`sets` fixes the circles and their order (by default, every event in `x` in
order of appearance). `scale` is pixels per panel width. `figure` and `axis`
are passed to `Figure` and `Axis`; other keywords go to [`vennplot`](@ref).

```julia
using PracticalProbability
@event A B
venn(P(A | B))                                            # one panel
venn(Equation(P(A ∪ B), inclusion_exclusion(P(A ∪ B))))  # P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
venn(Equation(P(A | B), bayes(P(A | B))))                 # Bayes' rule, as a fraction of panels
```
"""
venn(x; kwargs...) = _venn(_backend(), x; kwargs...)

"""
    vennplot(x; kwargs...)
    vennplot!(ax, x; kwargs...)

Draws the Venn panels of `x` into an axis: a `PracticalBayesPlot.Axis`, or
with Makie loaded a Makie axis (`vennplot` is then a Makie recipe). Use an
axis with `aspect = DataAspect()` to keep circles round. Attributes:

- `highlightcolor`: the event being measured, `E ∩ C` (default Makie blue);
- `givencolor`: the rest of the conditioning event `C` (default: a light tint
  of `highlightcolor`; unconditional panels are left white);
- `excludedcolor`: the part of the sample space ruled out by conditioning;
- `impossiblecolor`: hatching for impossible cells (`X = 1 ∩ X = 2`);
- `strokecolor`, `strokewidth`: circle and box outlines;
- `labels`: `:latex` (default), `:text` or `:none`, the term under each panel;
- `setlabels`: name each circle (default `true`);
- `textscale`: multiplies every font size;
- `backgroundcolor`: the unfilled sample space (default white); translucent
  fills are composited onto it, so the fill is opaque in every backend;
- `cellvalues`: with facts or samples, write cell probabilities in `:target`
  cells (default), `:all` cells, or `:none`;
- `pixels`: Makie only, pixels across each panel's fill (default 320); the
  PracticalBayesPlot drawing fills each region with an exact polygon.
"""
function vennplot end

"""
    vennplot!(ax, x; kwargs...)

Add the Venn panels of `x` to an existing axis. See [`vennplot`](@ref).
"""
function vennplot! end

# ---------------------------------------------------------------------------
# Which events become circles
# ---------------------------------------------------------------------------

_vennerr(msg) = throw(ArgumentError(string("no Venn diagram: ", msg)))

_vennpush!(out, a) = (a in out || push!(out, a); out)

function _vennbase!(out, a::RV)
    isevent(a) && return _vennpush!(out, a)
    _vennerr("P($(a.name)) is about the values of the variable $(a.name), not an event; " *
             "use events (@event A, $(a.name)(value), A ∪ B, …)")
end
_vennbase!(out, a::Event) = _vennpush!(out, a)
_vennbase!(out, a::Complement) = _vennbase!(out, a.event)
_vennbase!(out, a::Union{EventUnion,EventIntersection}) = (foreach(x -> _vennbase!(out, x), a.parts); out)
_vennbase!(out, ::Union{SureEvent,ImpossibleEvent}) = out
_vennbase!(out, a::Intervention) = _vennerr("$(a) is an intervention, not an event")
# Any other atom that is a simple event (e.g. an indexed event `A[i]`) is a
# circle; anything else (e.g. an atom added after this file was written) is not
# drawn. Describe it without assuming it can be printed.
_vennbase!(out, a::Atom) = _issimpleevent(a) ? _vennpush!(out, a) :
                           _vennerr("$(_vennstr(a)) is not an event a Venn diagram can show")
_issimpleevent(a) = !(a isa CompoundEvent) && !(a isa Intervention) && (try iseventlike(a) catch; false end)
_vennstr(a) = try
    sprint(show, a)
catch
    string(nameof(typeof(a)))
end

function _vennbase!(out, p::Prob)
    foreach(a -> _vennbase!(out, a), p.targets)
    foreach(a -> _vennbase!(out, a), p.given)
    return out
end
_vennbase!(out, ::Literal) = out
_vennbase!(out, e::Union{Product,Sum,Quotient}) = (foreach(c -> _vennbase!(out, c), children(e)); out)
_vennbase!(out, m::Marginal) =
    _vennerr("$(m) sums over the values of $(join((v.name for v in m.over), ", ")), which a Venn diagram cannot show")
# Expectations, plate products and other expression kinds have no Venn diagram.
_vennbase!(out, e::PExpr) = _vennerr("$(_vennstr(e)) is not a probability of events")
_vennbase!(out, e::Equation) = (_vennbase!(out, e.lhs); _vennbase!(out, e.rhs))
_vennbase!(out, d::Derivation) = (foreach(e -> _vennbase!(out, e), d.exprs); out)

"""
    venn_sets(x) -> Vector{Atom}

The events that [`venn`](@ref) draws as circles for `x`: every named event and
`X = value` in it, in order of first appearance. Throws an `ArgumentError` if
`x` has no Venn diagram (it involves plain random variables, `do(⋅)` or sums
over variables).
"""
venn_sets(x) = _vennbase!(Atom[], x)

# ---------------------------------------------------------------------------
# Cells and regions
# ---------------------------------------------------------------------------
#
# With n circles there are 2ⁿ cells; cell c (0-based) lies inside circle i
# exactly when bit i-1 of c is set (cell 0 is outside every circle). A region
# (an event) is a set of cells, stored as a bitmask over cells.

_ncells(n) = 1 << n
_allcells(n) = UInt32((UInt64(1) << _ncells(n)) - 1)
_incircle(c, i) = (c >> (i - 1)) & 1 == 1

function _circlecells(n, i)
    m = UInt32(0)
    for c in 0:_ncells(n)-1
        _incircle(c, i) && (m |= UInt32(1) << c)
    end
    return m
end

function _region(a::Atom, sets)
    n = length(sets)
    a isa SureEvent && return _allcells(n)
    a isa ImpossibleEvent && return UInt32(0)
    a isa Complement && return _allcells(n) & ~_region(a.event, sets)
    a isa EventUnion && return reduce(|, (_region(x, sets) for x in a.parts); init=UInt32(0))
    a isa EventIntersection && return reduce(&, (_region(x, sets) for x in a.parts); init=_allcells(n))
    i = findfirst(==(a), sets)
    i === nothing && throw(ArgumentError("$(a) is not one of the sets $(join(string.(sets), ", "))"))
    return _circlecells(n, i)
end
_region(atoms::AbstractVector, sets) =
    reduce(&, (_region(a, sets) for a in atoms); init=_allcells(length(sets)))

# Cells that cannot happen: inside two `X = value` circles of the same X with
# different values.
function _feasiblecells(sets)
    n = length(sets)
    m = _allcells(n)
    for i in 1:n, j in i+1:n
        a, b = sets[i], sets[j]
        if a isa Event && b isa Event && a.rv == b.rv && !isequal(a.value, b.value)
            m &= ~(_circlecells(n, i) & _circlecells(n, j))
        end
    end
    return m
end

# ---------------------------------------------------------------------------
# Geometry of one panel (unit coordinates, centred on the panel)
# ---------------------------------------------------------------------------

const VENN_BOX = (0.5, 0.36)          # half-width and half-height of the sample-space box

"""
    _venn_geometry(n) -> (centres, radius, labelpos)

Circle centres, the common radius and where each circle's name goes, for
`n = 0:3` circles, in panel coordinates (the box is `±0.5 × ±0.36`).
"""
function _venn_geometry(n::Integer)
    n == 0 && return (NTuple{2,Float64}[], 0.0, NTuple{2,Float64}[])
    n == 1 && return ([(0.0, 0.0)], 0.25, [(0.0, 0.31)])
    n == 2 && return ([(-0.15, 0.0), (0.15, 0.0)], 0.25, [(-0.33, 0.30), (0.33, 0.30)])
    n == 3 && return ([(-0.13, 0.1125), (0.13, 0.1125), (0.0, -0.1125)], 0.2,
                      [(-0.38, 0.27), (0.38, 0.27), (0.26, -0.27)])
    throw(ArgumentError("no Venn diagram: $n events; at most $VENN_MAXSETS can be drawn"))
end

"""
    _venn_cell(n, x, y) -> Int

The cell (0-based bitmask of circles) containing the point `(x, y)` of a panel.
"""
function _venn_cell(n, x, y)
    centres, r, _ = _venn_geometry(n)
    c = 0
    for (i, (cx, cy)) in enumerate(centres)
        hypot(x - cx, y - cy) < r && (c |= 1 << (i - 1))
    end
    return c
end

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

"""
    VennPanel

One Venn diagram in a [`VennLayout`](@ref): the term `prob` it shows, centred
at `(x, y)`. `target` and `given` are bitmasks over cells (see
[`venn_layout`](@ref)): `target` is `E ∩ C`, `given` is `C`, and
`conditional` says whether `C` is a real condition or the whole sample space.
"""
struct VennPanel
    prob::Prob
    target::UInt32
    given::UInt32
    conditional::Bool
    x::Float64
    y::Float64
end

"""
    VennText

A piece of text in a [`VennLayout`](@ref): an operator (`role = :op`), a
coefficient (`:coef`) or a derivation step's justification (`:note`).
"""
struct VennText
    text::String
    role::Symbol
    x::Float64
    y::Float64
    halign::Symbol
end

"""
    VennLine

A straight line in a [`VennLayout`](@ref), from `(x0, y0)` to `(x1, y1)`:
fraction bars and the strokes of brackets.
"""
struct VennLine
    x0::Float64
    y0::Float64
    x1::Float64
    y1::Float64
end

"""
    VennLayout

Everything [`venn`](@ref) draws, positioned: the events drawn as circles
(`sets`), which cells are impossible (`feasible` is the bitmask of the
possible ones), the [`VennPanel`](@ref)s, operators and other text
([`VennText`](@ref)), fraction bars and brackets ([`VennLine`](@ref)), and the
bounding box `(xmin, xmax, ymin, ymax)`. With facts or samples, `cellprobs[c + 1]`
is the probability of cell `c` (`nothing` where facts leave it open); without,
`cellprobs` is `nothing`. `conditions[i]` describes set `i` when it is a sample
event (e.g. `:(σ > 0.2)`), else `nothing`. Build one with [`venn_layout`](@ref).
"""
struct VennLayout
    sets::Vector{Atom}
    feasible::UInt32
    panels::Vector{VennPanel}
    texts::Vector{VennText}
    lines::Vector{VennLine}
    bbox::NTuple{4,Float64}
    cellprobs::Union{Nothing,Vector{Union{Nothing,Float64}}}
    conditions::Vector{Any}
end
VennLayout(sets, feasible, panels, texts, lines, bbox, cellprobs=nothing) =
    VennLayout(sets, feasible, panels, texts, lines, bbox, cellprobs, fill(nothing, length(sets)))

"""
    venn_value(L::VennLayout, panel::VennPanel) -> Union{Nothing,Float64}

The number a panel stands for, from the layout's cell probabilities: the
probability of its target cells, divided by that of its given cells when it is
conditional. `nothing` without facts, or when a cell it needs is undetermined.
"""
function venn_value(L::VennLayout, p::VennPanel)
    L.cellprobs === nothing && return nothing
    mass(mask) = begin
        s = 0.0
        for c in 0:length(L.cellprobs)-1
            mask & (UInt32(1) << c) == 0 && continue
            q = L.cellprobs[c+1]
            q === nothing && return nothing
            s += q
        end
        s
    end
    t = mass(p.target)
    t === nothing && return nothing
    p.conditional || return t
    g = mass(p.given)
    (g === nothing || g == 0) && return nothing
    return t / g
end

# Where to write a cell's number: the point of the cell farthest from every
# circle and from the box edge (found on a grid, once per n).
const _VENN_ANCHORS = Dict{Int,Vector{NTuple{2,Float64}}}()
function _venn_anchor(n::Integer, c::Integer)
    anchors = get!(_VENN_ANCHORS, n) do
        hw, hh = VENN_BOX
        centres, r, _ = _venn_geometry(n)
        best = fill((-Inf, (0.0, 0.0)), _ncells(n))
        for x in range(-hw, hw; length=161), y in range(-hh, hh; length=117)
            k = _venn_cell(n, x, y) + 1
            clear = min(hw - abs(x), hh - abs(y),
                        minimum((abs(hypot(x - cx, y - cy) - r) for (cx, cy) in centres); init=Inf))
            clear > best[k][1] && (best[k] = (clear, (x, y)))
        end
        [b[2] for b in best]
    end
    return anchors[c+1]
end

function Base.show(io::IO, L::VennLayout)
    print(io, "VennLayout(", length(L.sets), " sets: ", join(string.(L.sets), ", "),
          "; ", length(L.panels), length(L.panels) == 1 ? " panel)" : " panels)")
end

# A block under construction: its width, its extent above and below its centre
# line, and its items, positioned relative to (left edge, centre line).
struct _VBlock
    w::Float64
    top::Float64
    bot::Float64
    items::Vector{Any}
end

_vshift(p::VennPanel, dx, dy) = VennPanel(p.prob, p.target, p.given, p.conditional, p.x + dx, p.y + dy)
_vshift(t::VennText, dx, dy) = VennText(t.text, t.role, t.x + dx, t.y + dy, t.halign)
_vshift(l::VennLine, dx, dy) = VennLine(l.x0 + dx, l.y0 + dy, l.x1 + dx, l.y1 + dy)

const _VPANEL = (w=1.0, top=0.40, bot=0.54)   # box ±0.36 plus room for the label below
const _VOPW = 0.3                            # width of an operator
const _VGAP = 0.06                           # gap around fraction bars

function _vpanel(p::Prob, sets, feasible)
    n = length(sets)
    G = (isempty(p.given) ? _allcells(n) : _region(p.given, sets)) & feasible
    T = _region(p.targets, sets) & G
    return _VBlock(_VPANEL.w, _VPANEL.top, _VPANEL.bot,
                   Any[VennPanel(p, T, G, !isempty(p.given), _VPANEL.w / 2, 0.0)])
end

_vtext(s, role=:op; w=role === :op ? _VOPW : 0.1 + 0.06 * length(s)) =
    _VBlock(w, 0.1, 0.1, Any[VennText(s, role, w / 2, 0.0, :center)])

function _vhcat(blocks)
    items = Any[]
    x = 0.0
    for b in blocks
        append!(items, (_vshift(it, x, 0.0) for it in b.items))
        x += b.w
    end
    return _VBlock(x, maximum(b -> b.top, blocks; init=0.0), maximum(b -> b.bot, blocks; init=0.0), items)
end

function _vfrac(num::_VBlock, den::_VBlock)
    w = max(num.w, den.w) + 0.1
    items = Any[VennLine(0.0, 0.0, w, 0.0)]
    dyn = _VGAP + num.bot
    dyd = -(_VGAP + den.top)
    append!(items, (_vshift(it, (w - num.w) / 2, dyn) for it in num.items))
    append!(items, (_vshift(it, (w - den.w) / 2, dyd) for it in den.items))
    return _VBlock(w, dyn + num.top, -dyd + den.bot, items)
end

# Square brackets around a block, as in the text form `[P(B | A) P(A)] / P(B)`.
function _vbracket(b::_VBlock)
    pad, arm = 0.12, 0.05
    w = b.w + 2pad
    y0, y1 = -b.bot + 0.12, b.top
    items = Any[VennLine(pad / 2, y0, pad / 2, y1), VennLine(pad / 2, y0, pad / 2 + arm, y0),
                VennLine(pad / 2, y1, pad / 2 + arm, y1),
                VennLine(w - pad / 2, y0, w - pad / 2, y1), VennLine(w - pad / 2, y0, w - pad / 2 - arm, y0),
                VennLine(w - pad / 2, y1, w - pad / 2 - arm, y1)]
    append!(items, (_vshift(it, pad, 0.0) for it in b.items))
    return _VBlock(w, b.top, b.bot, items)
end

_vnum(v) = v isa Integer || (v isa AbstractFloat && isinteger(v) && abs(v) < 1e15) ? string(Integer(v)) :
           v isa Rational ? string(numerator(v), "/", denominator(v)) : string(v)

_vblock(p::Prob, sets, F) = _vpanel(p, sets, F)
_vblock(l::Literal, sets, F) = _vtext(_vnum(l.value), :coef)

function _vblock(s::Sum, sets, F)
    parts = _VBlock[]
    for (i, t) in enumerate(s.terms)
        neg, a = _signed(t)
        (i > 1 || neg) && push!(parts, _vtext(neg ? "−" : "+", :op; w=i == 1 ? 0.18 : _VOPW))
        b = _vblock(a, sets, F)
        push!(parts, a isa Sum ? _vbracket(b) : b)
    end
    return _vhcat(parts)
end

function _vblock(p::Product, sets, F)
    parts = _VBlock[]
    fs = p.factors
    start = 1
    if fs[1] isa Literal
        push!(parts, fs[1].value == -1 ? _vtext("−", :op; w=0.18) : _vtext(_vnum(fs[1].value), :coef))
        start = 2
    end
    for i in start:length(fs)
        i > start && push!(parts, _vtext("×", :op))
        f = fs[i]
        b = _vblock(f, sets, F)
        push!(parts, f isa Sum ? _vbracket(b) : b)
    end
    return _vhcat(parts)
end

_vblock(q::Quotient, sets, F) = _vfrac(_vblock(q.num, sets, F), _vblock(q.den, sets, F))
_vblock(m::Marginal, sets, F) = _vennbase!(Atom[], m)       # throws

_vblock(e::Equation, sets, F) =
    _vhcat([_vblock(e.lhs, sets, F), _vtext(e.relation === :propto ? "∝" : "="), _vblock(e.rhs, sets, F)])

# A derivation: one row per step, "=" signs in a column, justifications on the right.
function _vblock(d::Derivation, sets, F)
    rows = _VBlock[]
    for (i, e) in enumerate(d.exprs)
        lead = i == 1 ? _VBlock(_VOPW, 0.0, 0.0, Any[]) : _vtext("=")
        push!(rows, _vhcat([lead, _vblock(e, sets, F)]))
    end
    notex = maximum(r -> r.w, rows) + 0.15
    items = Any[]
    y = 0.0
    for (i, r) in enumerate(rows)
        i > 1 && (y -= r.top)
        append!(items, (_vshift(it, 0.0, y) for it in r.items))
        i > 1 && push!(items, VennText(d.rules[i], :note, notex, y, :left))
        y -= r.bot + 0.08
    end
    notew = maximum((0.055 * length(d.rules[i]) for i in 2:length(d)); init=0.0)
    return _VBlock(notex + notew, rows[1].top, -(y + 0.08), items)
end

"""
    venn_layout(x; sets = venn_sets(x)) -> VennLayout

Work out the Venn diagrams of `x` (a probability expression about events, an
[`Equation`](@ref) or a [`Derivation`](@ref)) without drawing them: the circles,
the highlighted cells of each panel and the position of every panel, operator
and fraction bar. [`venn`](@ref) draws the result.

Cells are numbered `0:2ⁿ-1` for `n` circles; cell `c` lies inside circle `i`
exactly when bit `i-1` of `c` is set. A panel's `target` and `given` are
bitmasks over those cells.

`facts` (the same facts [`deduce`](@ref) takes: `P(A) => 0.3`, `A ⟂ B`,
`disjoint(A, B)`, …, or a `Facts`) fills in the probability of every cell they
determine, via [`cell_probabilities`](@ref). [`venn`](@ref) then writes those
numbers in the cells and each panel's value under it. This needs the circles to
be named events.

`samples` ([`SampleEvents`](@ref), e.g. from [`@sampleevents`](@ref) on posterior
draws) does the same with frequencies over the draws, and labels each circle
with its condition (`σ > 0.2`).
"""
function venn_layout(x; sets=nothing, facts=nothing, samples=nothing)
    facts === nothing || samples === nothing || throw(ArgumentError("give facts or samples, not both"))
    found = venn_sets(x)
    S = sets === nothing ? found : _atoms(sets)
    sets === nothing || all(a -> a in S, found) ||
        throw(ArgumentError("sets must include every event in the expression: $(join(string.(found), ", "))"))
    for a in S
        _issimpleevent(a) ||
            throw(ArgumentError("each set must be a named event or X(value), got $(a)"))
    end
    length(S) <= VENN_MAXSETS ||
        _vennerr("$(length(S)) events ($(join(string.(S), ", "))); at most $VENN_MAXSETS can be drawn")
    F = _feasiblecells(S)
    b = _vblock(x, S, F)
    panels = VennPanel[it for it in b.items if it isa VennPanel]
    texts = VennText[it for it in b.items if it isa VennText]
    lines = VennLine[it for it in b.items if it isa VennLine]
    src = samples === nothing ? facts : samples
    return VennLayout(S, F, panels, texts, lines, (0.0, b.w, -b.bot, b.top), _venn_cellprobs(S, src),
                      _venn_conditions(S, samples))
end
function venn_layout(L::VennLayout; sets=nothing, facts=nothing, samples=nothing)
    src = samples === nothing ? facts : samples
    src === nothing && return L
    return VennLayout(L.sets, L.feasible, L.panels, L.texts, L.lines, L.bbox, _venn_cellprobs(L.sets, src),
                      _venn_conditions(L.sets, samples))
end

_venn_conditions(S, ::Nothing) = Any[nothing for _ in S]
_venn_conditions(S, se) = Any[(k = findfirst(==(a), se.events); k === nothing ? nothing : se.conditions[k]) for a in S]

_venn_cellprobs(S, ::Nothing) = nothing
function _venn_cellprobs(S, facts)
    all(a -> a isa RV && isevent(a), S) ||
        throw(ArgumentError("cell probabilities from facts need named events (@event A) as the circles"))
    fs = facts isa Union{Tuple,AbstractVector} ? facts : (facts,)   # facts, or one SampleEvents
    # The facts may mention events that are not circles: include them, then sum them out.
    own, _ = cell_probabilities(fs...)
    evs, probs = cell_probabilities(fs...; events=vcat(RV[a for a in S], RV[e for e in own if !(e in S)]))
    perm = [findfirst(==(a), evs) for a in S]      # where each circle sits in evs
    out = Union{Nothing,Float64}[0.0 for _ in 1:_ncells(length(S))]
    for k in 0:length(probs)-1
        c = 0
        for (i, j) in enumerate(perm)
            (k >> (j - 1)) & 1 == 1 && (c |= 1 << (i - 1))
        end
        out[c+1] === nothing && continue
        out[c+1] = probs[k+1] === nothing ? nothing : out[c+1] + probs[k+1]
    end
    return out
end

# Labels: the LaTeX of a term, restricted to what Makie's MathTeXEngine renders.
_venn_latex(x) = replace(_latex(x), "^{\\mathsf{c}}" => "^{c}", "\\varnothing" => "\\emptyset")

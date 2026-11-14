# The diagram functions. Layouts are computed in pure Julia (dagplot.jl,
# probtree.jl, venn.jl); the drawing has two backends:
#
# - PracticalBayesPlot (`:pbp`, the default, src/draw/): light, always loaded,
#   draws SVG (shown inline in VS Code, Jupyter, Pluto, Quarto) and TikZ;
# - Makie (`:makie`, ext/PracticalProbabilityMakieExt.jl), once a Makie
#   backend such as CairoMakie is loaded.
#
# The figure-level functions (`dagplot(g)`, `venn(x)`, ...) dispatch on
# `plotbackend()`; the `!` forms dispatch on the type of the axis given.

const _BACKEND = Ref(:pbp)
const _BACKEND_CHOSEN = Ref(false)
const _MAKIE_LOADED = Ref(false)

"""
    plotbackend() -> Symbol

The backend the figure-level diagram functions ([`dagplot`](@ref),
[`venn`](@ref), [`probtree`](@ref), [`mosaic`](@ref), [`factorplot`](@ref))
draw with: `:pbp` (PracticalBayesPlot, the default) or `:makie`. Loading a
Makie backend (`using CairoMakie`) switches to `:makie`, unless
[`plotbackend!`](@ref) chose one first.
"""
plotbackend() = _BACKEND[]

"""
    plotbackend!(backend::Symbol)

Choose what the figure-level diagram functions draw with:

- `:pbp`: PracticalBayesPlot, always available. Figures are
  `PracticalBayesPlot.Figure`s, which display as SVG and can be written as
  SVG or as TikZ for LaTeX (`PracticalBayesPlot.save("f.tex", fig)`).
- `:makie`: Makie (needs a Makie backend such as CairoMakie loaded).
  Figures are `Makie.Figure`s.

The choice holds for the session; loading Makie afterwards does not change it.
"""
function plotbackend!(b::Symbol)
    b in (:pbp, :makie) || throw(ArgumentError("plot backend must be :pbp or :makie, got :$b"))
    b === :makie && !_MAKIE_LOADED[] &&
        throw(ArgumentError("the :makie backend needs a Makie backend package loaded first (`using CairoMakie`)"))
    _BACKEND[] = b
    _BACKEND_CHOSEN[] = true
    return b
end

# called by the Makie extension when it loads
function _makie_loaded!()
    _MAKIE_LOADED[] = true
    _BACKEND_CHOSEN[] || (_BACKEND[] = :makie)
    return nothing
end

_backend() = Val(_BACKEND[])

# backend methods: _name(::Val{:pbp}, ...) in src/draw/, _name(::Val{:makie}, ...) in the extension
function _dagplot end
function _probtree end
function _mosaic end
function _factorplot end
function _venn end

"""
    dagplot(g::DAG; plates = nothing, kwargs...)
    dagplot(g::DAG, A ⟂ C | B; kwargs...)
    dagplot(g::DAG, P(Y | Do(X)); adjust = nothing, kwargs...)
    dagplot(m::GenerativeModel; kwargs...)

Draw a DAG and return the figure (see [`plotbackend`](@ref)). The layout
comes from [`dag_view`](@ref); `dagplot!(ax, …)` draws into an existing axis.

- With an independence statement, every path between the two sides is drawn
  and classified: open paths in red, blocked paths dashed grey, with the node
  that blocks each one ringed. The conditioning set is shaded, and the caption
  says whether the statement holds.
- With an interventional query `P(Y | Do(X))`, there are two panels. On the
  left, the back-door paths from `X` to `Y` and the adjustment set that closes
  them (from [`identify`](@ref), or `adjust`). On the right, the graph after
  surgery: the edges into `X` are cut. The caption is `identify`'s answer.
- A [`GenerativeModel`](@ref) (from `@graphical`, `@generative` or
  [`generative`](@ref)) is drawn in plate notation. Observed variables are
  shaded, inputs are small dots, and repeated variables sit in labelled plates.
- `plates = Dict(:i => [x, y], …)` draws plates on a plain DAG.

Keywords: `scale` (pixels per unit; nodes are one unit apart), `figure`,
`axis`, and the colours `highlightcolor` (endpoints and adjustment set),
`opencolor`, `blockedcolor`, `observedcolor`.
"""
dagplot(x, q=nothing; kwargs...) = _dagplot(_backend(), x, q; kwargs...)

"""
    dagplot!(ax, x; kwargs...)

Draw the DAG view of `x` into an existing axis (a `PracticalBayesPlot.Axis`
or, with Makie loaded, a Makie axis). See [`dagplot`](@ref).
"""
function dagplot! end

"""
    probtree(x; order = nothing, bn = nothing, facts = nothing, dag = nothing, kwargs...)

Draw a probability tree and return the figure (see [`plotbackend`](@ref)).
`x` is either a list of variables or events (the levels of the tree,
root first) or a query `P(E | C)` about events.

Each level branches on one variable: an event `A` into `A` and `Aᶜ`, a discrete
variable into its values (from `bn`). Each branch is labelled with its
conditional probability `P(branch | path so far)`, simplified with `dag` (or
`bn`'s graph) when given. With numbers from a [`BayesNet`](@ref) `bn` or from
`facts` (as for [`deduce`](@ref)), branches carry their values and leaves their
joint probabilities.

For a query `P(E | C)` the leaves are coloured as in [`venn`](@ref): leaves in
`E ∩ C` filled, the rest of `C` tinted, the others greyed, and the caption
works out `P(E | C) = P(E ∩ C) / P(C)` from the leaves. The layout comes from
[`tree_layout`](@ref); `probtree!(ax, …)` draws into an existing axis.
"""
probtree(x; kwargs...) = _probtree(_backend(), x; kwargs...)

"""
    probtree!(ax, x; kwargs...)

Draw a probability tree into an existing axis. See [`probtree`](@ref).
"""
function probtree! end

"""
    mosaic(x; order = nothing, bn = nothing, facts = nothing, kwargs...)

Draw a mosaic plot (an area-proportional diagram) and return the figure
(see [`plotbackend`](@ref)). Arguments are as for
[`probtree`](@ref), but numbers are required: the unit square is split by the
first variable, each strip by the second given the first, and so on, so that
every tile's area is its joint probability.

For a query `P(E | C)` the tiles are coloured as in [`venn`](@ref), so the
answer is literally the blue area divided by the blue and tinted area. Unlike a
Venn diagram, the areas are to scale. The layout comes from
[`mosaic_layout`](@ref); `mosaic!(ax, …)` draws into an existing axis.
"""
mosaic(x; kwargs...) = _mosaic(_backend(), x; kwargs...)

"""
    mosaic!(ax, x; kwargs...)

Draw a mosaic plot into an existing axis. See [`mosaic`](@ref).
"""
function mosaic! end

"""
    factorplot(f::Factor; kind = :bars, kwargs...)

Plot a [`Factor`](@ref) (for example the result of [`compute`](@ref)) and
return the figure (see [`plotbackend`](@ref)).

- One variable: a bar per value.
- Two variables: grouped bars, the first variable along the axis and one colour
  per value of the second (for a conditional table such as `P(A | B)`, one
  group of bars per condition). `kind = :heatmap` draws a heatmap with the
  values written in the cells instead.
- Three variables: one panel per value of the third.

With Makie, `barplot(f)` and `heatmap(f)` also work on one- and
two-variable factors. `factorplot!(ax, f)` draws into an existing axis (one
or two variables).
"""
factorplot(f; kwargs...) = _factorplot(_backend(), f; kwargs...)

"""
    factorplot!(ax, f::Factor; kind = :bars, kwargs...)

Plot a one- or two-variable [`Factor`](@ref) into an existing axis. See
[`factorplot`](@ref).
"""
function factorplot! end

module PracticalProbabilityMakieExt

# Draws the Venn layouts computed in src/venn.jl. Each panel's fill is a small
# image: every pixel is a mix of the colours of the cells it touches, weighted
# by how much of the pixel each circle covers. The weights over the 2ⁿ cells
# sum to one, which gives anti-aliased region edges for any combination of
# cells. Outlines, operators and labels are drawn as vectors on top.

using Makie: Makie
using PracticalProbability: PracticalProbability, LaTeXString, PExpr, Equation, Derivation, Factor,
                            VennLayout, VennPanel, venn_layout, DAGView, dag_view, TreeLayout, tree_layout,
                            MosaicLayout, mosaic_layout
import PracticalProbability: vennplot, vennplot!, dagplot!, probtree!, mosaic!, factorplot!

const PP = PracticalProbability

Makie.@recipe VennPlot (layout,) begin
    "Fill of the event being measured, `E ∩ C` for a panel `P(E | C)`."
    highlightcolor = Makie.wong_colors()[1]
    "Fill of the rest of the conditioning event `C`; `nothing` uses a light tint of `highlightcolor`."
    givencolor = nothing
    "Fill of the sample space ruled out by conditioning."
    excludedcolor = (:gray55, 0.4)
    "Hatching of impossible cells (`X = 1 ∩ X = 2`)."
    impossiblecolor = (:gray40, 0.45)
    "Colour of outlines, fraction bars and brackets."
    strokecolor = :black
    "Width of outlines, fraction bars and brackets."
    strokewidth = 1.5
    "Colour of all text."
    textcolor = :black
    "The term under each panel: `:latex`, `:text` or `:none`."
    labels = :latex
    "Name each circle."
    setlabels = true
    "Multiplies every font size."
    textscale = 1.0
    "Colour of the sample space where nothing is filled in; the fill is composited onto it."
    backgroundcolor = :white
    "Pixels across each panel's fill."
    pixels = 320
    "With facts or samples: write cell probabilities in `:target` cells (default), `:all` cells, or `:none`."
    cellvalues = :target
    "Colour of cell probabilities written on the highlight colour."
    cellvaluecolor = :white
end

for T in (PExpr, Equation, Derivation)
    @eval Makie.convert_arguments(::Type{<:VennPlot}, x::$T) = (venn_layout(x),)
end

# Font sizes, in data units (a panel is one unit wide).
const FONTSIZE = (op=0.16, coef=0.14, note=0.1, label=0.075, set=0.068, cell=0.058)

_fmt(x::Real) = string(round(x; sigdigits=3))
_fmt(::Nothing) = "?"

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

# LaTeX if MathTeXEngine can lay it out, the plain-text form otherwise.
function _tex(s::AbstractString, fallback::AbstractString)
    L = LaTeXString(string("\$", s, "\$"))
    isdefined(Makie, :MathTeXEngine) || return L
    try
        Makie.MathTeXEngine.generate_tex_elements(L)
        return L
    catch
        return fallback
    end
end

function _label(x, mode::Symbol; suffix="")
    mode === :none && return ""
    plain = sprint(show, x) * suffix
    mode === :text && return plain
    mode === :latex && return _tex(PP._venn_latex(x) * suffix, plain)
    throw(ArgumentError("labels must be :latex, :text or :none, got :$mode"))
end

# A circle's name: its condition for sample events (`σ > 0.2`), else the event.
function _setlabel(a, cond, mode::Symbol)
    cond === nothing && return _label(a, mode === :none ? :latex : mode)
    plain = string(cond)
    mode === :text && return plain
    lx = PP._conditionlatex(cond)
    return lx === nothing ? plain : _tex(lx, plain)
end

# ---------------------------------------------------------------------------
# Panel fill
# ---------------------------------------------------------------------------

_rgba(c) = Makie.to_color(c)

"""
    _panel_image(n, panel, feasible, colours; pixels, background) -> (img, (x0, x1), (y0, y1))

The fill of one panel as an opaque RGBA matrix (`img[i, j]`: column `i` from
the left, row `j` from the bottom), with the panel-relative extent it covers.
Translucent colours are composited onto `background` here rather than left to
the backend: vector backends (PDF, SVG) draw per-pixel alpha unreliably.
"""
function _panel_image(n::Integer, panel::VennPanel, feasible::UInt32, colours;
                      pixels::Integer=320, background=Makie.RGBAf(1, 1, 1, 1))
    hw, hh = PP.VENN_BOX
    centres, r, _ = PP._venn_geometry(n)
    nx = pixels
    ny = max(1, round(Int, pixels * hh / hw))
    px = 2hw / nx
    clear = Makie.RGBAf(1, 1, 1, 0)
    ncells = PP._ncells(n)
    img = Matrix{Makie.RGBAf}(undef, nx, ny)
    cov = zeros(Float32, n)
    for j in 1:ny, i in 1:nx
        x = -hw + (i - 0.5) * px
        y = -hh + (j - 0.5) * px
        for (k, (cx, cy)) in enumerate(centres)
            cov[k] = clamp(0.5f0 - Float32((hypot(x - cx, y - cy) - r) / px), 0f0, 1f0)
        end
        hatch = mod((x + y) / 0.045, 1.0) < 0.5
        R = G = B = A = 0f0
        for c in 0:ncells-1
            w = 1f0
            for k in 1:n
                w *= PP._incircle(c, k) ? cov[k] : 1f0 - cov[k]
            end
            w == 0 && continue
            bit = UInt32(1) << c
            col = if feasible & bit == 0
                hatch ? colours.impossible : clear
            elseif panel.target & bit != 0
                colours.highlight
            elseif panel.given & bit != 0
                panel.conditional ? colours.given : clear
            else
                colours.excluded
            end
            a = w * Makie.alpha(col)
            R += a * Makie.red(col)
            G += a * Makie.green(col)
            B += a * Makie.blue(col)
            A += a
        end
        # R, G, B are premultiplied by alpha: composite over the background
        bg = background
        img[i, j] = Makie.RGBAf(R + (1 - A) * Makie.red(bg), G + (1 - A) * Makie.green(bg),
                                B + (1 - A) * Makie.blue(bg), 1)
    end
    return img, (-hw, hw), (-hh, hh)
end

_circle(cx, cy, r; n=200) = [Makie.Point2f(cx + r * cospi(2t / n), cy + r * sinpi(2t / n)) for t in 0:n]

function _draw_panel!(p, L::VennLayout, panel::VennPanel, colours, style)
    n = length(L.sets)
    img, (x0, x1), (y0, y1) = _panel_image(n, panel, L.feasible, colours; pixels=style.pixels,
                                           background=style.background)
    X, Y = panel.x, panel.y
    Makie.image!(p, Makie.IntervalSets.ClosedInterval(X + x0, X + x1),
                 Makie.IntervalSets.ClosedInterval(Y + y0, Y + y1), img; interpolate=true)
    centres, r, labelpos = PP._venn_geometry(n)
    for (cx, cy) in centres
        Makie.lines!(p, _circle(X + cx, Y + cy, r); color=style.stroke, linewidth=style.width)
    end
    Makie.lines!(p, Makie.Point2f[(X + x0, Y + y0), (X + x1, Y + y0), (X + x1, Y + y1),
                                  (X + x0, Y + y1), (X + x0, Y + y0)];
                 color=style.stroke, linewidth=style.width)
    if style.setlabels
        for (a, cond, (lx, ly)) in zip(L.sets, L.conditions, labelpos)
            Makie.text!(p, Makie.Point2f(X + lx, Y + ly); text=_setlabel(a, cond, style.labels),
                        fontsize=FONTSIZE.set * style.scale, markerspace=:data,
                        align=(:center, :center), color=style.text)
        end
    end
    if L.cellprobs !== nothing && style.cellvalues !== :none
        style.cellvalues in (:target, :all) ||
            throw(ArgumentError("cellvalues must be :target, :all or :none, got :$(style.cellvalues)"))
        for c in 0:PP._ncells(n)-1
            bit = UInt32(1) << c
            L.feasible & bit == 0 && continue
            intarget = panel.target & bit != 0
            style.cellvalues === :target && !intarget && continue
            ax_, ay_ = PP._venn_anchor(n, c)
            Makie.text!(p, Makie.Point2f(X + ax_, Y + ay_); text=_fmt(L.cellprobs[c+1]),
                        fontsize=FONTSIZE.cell * style.scale, markerspace=:data, align=(:center, :center),
                        color=intarget ? style.cellcolor : style.text)
        end
    end
    v = PP.venn_value(L, panel)
    lab = _label(panel.prob, style.labels; suffix=v === nothing ? "" : " = " * _fmt(v))
    isempty(lab) || Makie.text!(p, Makie.Point2f(X, Y - PP.VENN_BOX[2] - 0.09); text=lab,
                                fontsize=FONTSIZE.label * style.scale, markerspace=:data,
                                align=(:center, :center), color=style.text)
    return p
end

# ---------------------------------------------------------------------------
# The recipe
# ---------------------------------------------------------------------------

function Makie.plot!(p::VennPlot)
    L = venn_layout(p[1][])
    hl = _rgba(p.highlightcolor[])
    gc = p.givencolor[]
    colours = (highlight=hl,
               given=gc === nothing ? Makie.RGBAf(Makie.red(hl), Makie.green(hl), Makie.blue(hl), 0.18) : _rgba(gc),
               excluded=_rgba(p.excludedcolor[]),
               impossible=_rgba(p.impossiblecolor[]))
    style = (stroke=p.strokecolor[], width=p.strokewidth[], text=p.textcolor[], labels=p.labels[],
             setlabels=p.setlabels[], scale=p.textscale[], pixels=p.pixels[],
             background=_rgba(p.backgroundcolor[]), cellvalues=p.cellvalues[],
             cellcolor=p.cellvaluecolor[])
    for panel in L.panels
        _draw_panel!(p, L, panel, colours, style)
    end
    for l in L.lines
        Makie.lines!(p, Makie.Point2f[(l.x0, l.y0), (l.x1, l.y1)]; color=style.stroke, linewidth=style.width)
    end
    for t in L.texts
        Makie.text!(p, Makie.Point2f(t.x, t.y); text=t.text, fontsize=getfield(FONTSIZE, t.role) * style.scale,
                    markerspace=:data, align=(t.halign, :center), color=style.text)
    end
    return p
end

function PP._venn(::Val{:makie}, x; sets=nothing, facts=nothing, samples=nothing, scale::Real=260,
                  figure=NamedTuple(), axis=NamedTuple(), kwargs...)
    L = venn_layout(x; sets, facts, samples)
    xmin, xmax, ymin, ymax = L.bbox
    m = 0.06
    W, H = xmax - xmin + 2m, ymax - ymin + 2m
    fig = Makie.Figure(; size=(round(Int, scale * W) + 32, round(Int, scale * H) + 32), figure...)
    ax = Makie.Axis(fig[1, 1]; aspect=Makie.DataAspect(), axis...)
    pl = vennplot!(ax, L; kwargs...)
    Makie.hidedecorations!(ax)
    Makie.hidespines!(ax)
    Makie.limits!(ax, xmin - m, xmax + m, ymin - m, ymax + m)
    return Makie.FigureAxisPlot(fig, ax, pl)
end

# ---------------------------------------------------------------------------
# The other diagrams: plain functions drawing into an axis (`name!(ax, …)`) and
# a figure-level `name(…)` that sizes the figure to the layout.
# ---------------------------------------------------------------------------

# A figure of the right size for a layout with bounding box `bbox`, with one
# borderless, equal-aspect axis.
function _figure(bbox, scale, figure, axis; m=0.1)
    xmin, xmax, ymin, ymax = bbox
    W, H = xmax - xmin + 2m, ymax - ymin + 2m
    fig = Makie.Figure(; size=(round(Int, scale * W) + 32, round(Int, scale * H) + 32), figure...)
    ax = Makie.Axis(fig[1, 1]; aspect=Makie.DataAspect(), axis...)
    Makie.hidedecorations!(ax)
    Makie.hidespines!(ax)
    Makie.limits!(ax, xmin - m, xmax + m, ymin - m, ymax + m)
    return fig, ax
end

_tint(c, a) = (c = _rgba(c); Makie.RGBAf(Makie.red(c), Makie.green(c), Makie.blue(c), a))
# a translucent colour composited onto white (opaque, so every backend draws it the same)
_onwhite(c, a) = (c = _rgba(c); Makie.RGBAf(1 - a + a * Makie.red(c), 1 - a + a * Makie.green(c), 1 - a + a * Makie.blue(c), 1))

const ROLECOLOURS = (target=1.0, given=0.18)

include("makie/dag.jl")
include("makie/tree.jl")
include("makie/factor.jl")

# figure-level diagram functions draw with Makie from now on (unless the user chose)
__init__() = PP._makie_loaded!()

end

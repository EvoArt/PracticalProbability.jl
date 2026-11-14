# Drawing the diagram layouts with PracticalBayesPlot (the default backend;
# see plots.jl). These mirror the Makie drawings in
# ext/PracticalProbabilityMakieExt.jl: same layouts, sizes (in data units)
# and colours. Text carries a plain form, shown in SVG, and a LaTeX form,
# which TikZ output typesets.

import PracticalBayesPlot as PBP

const _WHITE = (1.0, 1.0, 1.0)

_rgb3(c) = (r = PBP.rgba(c); (r[1], r[2], r[3]))

# a translucent colour composited onto white (opaque, as in the Makie drawings)
_pwhite(c, a) = (r = PBP.rgba(c); a *= r[4]; (1 - a + a * r[1], 1 - a + a * r[2], 1 - a + a * r[3]))

# `fg` with opacity `a` over the opaque `bg`
_over(fg::NTuple{3,Float64}, a::Real, bg::NTuple{3,Float64}) =
    (a * fg[1] + (1 - a) * bg[1], a * fg[2] + (1 - a) * bg[2], a * fg[3] + (1 - a) * bg[3])

# (plain, latex) forms of a label; mode :latex, :text or :none
function _plabel(x, mode::Symbol; suffix::AbstractString="")
    mode === :none && return ("", "")
    plain = sprint(show, x) * suffix
    mode === :text && return (plain, "")
    mode === :latex || throw(ArgumentError("labels must be :latex, :text or :none, got :$mode"))
    lx = try
        _venn_latex(x)
    catch
        ""
    end
    return (plain, isempty(lx) ? "" : lx * suffix)
end

_plainnum(x::Real) = string(round(x; sigdigits=3))
_plainnum(::Nothing) = "?"

# text in data units
function _ptext!(ax, x, y, (plain, latex)::Tuple{AbstractString,AbstractString}; fontsize, align=(:center, :center),
                 color=:black)
    isempty(plain) && return
    PBP.text!(ax, Float64(x), Float64(y); text=plain, latex=isempty(latex) ? nothing : latex, fontsize,
              markerspace=:data, align, color)
end
_ptext!(ax, x, y, s::AbstractString; kwargs...) = _ptext!(ax, x, y, (s, ""); kwargs...)

_pcircle(cx, cy, r; n=96) = [(cx + r * cospi(2t / n), cy + r * sinpi(2t / n)) for t in 0:n-1]

# A figure sized to a layout's bounding box, with one equal-aspect axis and
# no decorations: `scale` pixels per data unit, as in the Makie drawings.
function _pfigure(bbox, scale, figure, axis; m=0.1)
    xmin, xmax, ymin, ymax = bbox
    W, H = xmax - xmin + 2m, ymax - ymin + 2m
    fig = PBP.Figure(; size=(round(Int, scale * W) + 20, round(Int, scale * H) + 20), figure...)
    ax = PBP.Axis(fig[1, 1]; aspect=PBP.DataAspect(), limits=(xmin - m, xmax + m, ymin - m, ymax + m), axis...)
    PBP.hidedecorations!(ax)
    PBP.hidespines!(ax)
    return fig, ax
end

# A shared renderer with a small primitive adapter for each backend.
_mclines!(ax::PBP.Axis, ps; kwargs...) = PBP.lines!(ax, first.(ps), last.(ps); kwargs...)
_mcpoly!(ax::PBP.Axis, ps; kwargs...) = PBP.poly!(ax, ps; kwargs...)
_mctext!(ax::PBP.Axis, p, label; kwargs...) = PBP.text!(ax, p...; text=label, markerspace=:data, kwargs...)

function markovplot!(ax, v::MarkovView; labels=true, color=:steelblue, absorbingcolor=:gray82,
                     textcolor=:black, linewidth=1.5)
    for e in v.transitions
        _mclines!(ax, e.path; color, linewidth)
        _mcpoly!(ax, e.head; color)
        labels && _mctext!(ax, e.labelpos, _numstr(e.probability); fontsize=0.16,
                          align=(:center, :center), color=textcolor)
    end
    for (i, (x, y)) in enumerate(v.positions)
        _mcpoly!(ax, _pcircle(x, y, 0.3); color=v.absorbing[i] ? absorbingcolor : :white,
                 strokecolor=:black, strokewidth=linewidth)
        label = string(v.states[i])
        _mctext!(ax, (x, y), label; fontsize=0.22min(1.0, 4/max(1, length(label))),
                 align=(:center, :center), color=textcolor)
    end
    return v
end

function _markovplot(::Val{:pbp}, v::MarkovView; scale=140, figure=(;), axis=(;), kwargs...)
    fig, ax = _pfigure(v.bbox, scale, figure, axis)
    markovplot!(ax, v; kwargs...)
    return fig
end

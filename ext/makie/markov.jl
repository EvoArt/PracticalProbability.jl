PP._mclines!(ax::Makie.Axis, ps; kwargs...) = Makie.lines!(ax, Makie.Point2f.(ps); kwargs...)
PP._mcpoly!(ax::Makie.Axis, ps; kwargs...) = Makie.poly!(ax, Makie.Point2f.(ps); kwargs...)
PP._mctext!(ax::Makie.Axis, p, label; kwargs...) =
    Makie.text!(ax, Makie.Point2f(p); text=label, markerspace=:data, kwargs...)

function PP._markovplot(::Val{:makie}, v::PP.MarkovView; scale=140, figure=(;), axis=(;), kwargs...)
    fig, ax = _figure(v.bbox, scale, figure, axis)
    PP.markovplot!(ax, v; kwargs...)
    return fig
end

# Plots of Factors (results of `compute`) with PracticalBayesPlot: bars,
# grouped bars, heatmaps.

# Tick labels for a variable's values: an event's [true, false] as A, Aᶜ.
function _pvaluelabels(v, dom)
    if isevent(v) && length(dom) == 2 && all(x -> x isa Bool, dom)
        return [d ? string(v.name) : string(v.name, "ᶜ") for d in dom]
    end
    return [string(d) for d in dom]
end

function factorplot!(ax::PBP.Axis, f::Factor; kind::Symbol=:bars, color=1, palette=PBP.wong_colors(),
                     colormap=:viridis, values::Bool=true, legend::Bool=true)
    n = length(f.vars)
    n in (1, 2) || throw(ArgumentError("factorplot! draws factors of one or two variables; " *
                                       "this one has $(n). Use factorplot, or sumout first"))
    v1, d1 = f.vars[1], f.domains[1]
    xs = collect(1.0:length(d1))
    ax.xticks = (xs, _pvaluelabels(v1, d1))
    ax.xlabel = string(v1.name)
    ax.xgridvisible = false
    if n == 1
        PBP.barplot!(ax, xs, vec(f.values); color)
        ax.ylabel = f.label
        return ax
    end
    v2, d2 = f.vars[2], f.domains[2]
    if kind === :heatmap
        ys = collect(1.0:length(d2))
        PBP.heatmap!(ax, xs, ys, f.values; colormap)
        ax.yticks = (ys, _pvaluelabels(v2, d2))
        ax.ylabel = string(v2.name)
        ax.ygridvisible = false
        if values
            lo, hi = extrema(f.values)
            for i in eachindex(xs), j in eachindex(ys)
                x = f.values[i, j]
                PBP.text!(ax, xs[i], ys[j]; text=_plainnum(x), align=(:center, :center),
                          color=(hi > lo && (x - lo) / (hi - lo) > 0.6) ? :black : :white)
            end
        end
        return ax
    elseif kind === :bars
        n2 = length(d2)
        w = 0.8 / n2
        labels = _pvaluelabels(v2, d2)
        for g in 1:n2
            PBP.barplot!(ax, xs .+ (g - (n2 + 1) / 2) * w, f.values[:, g]; width=w,
                         color=palette[mod1(g, length(palette))], label=labels[g])
        end
        ax.ylabel = f.label
        legend && PBP.axislegend(ax, string(v2.name))      # where it covers the fewest bars
        return ax
    end
    throw(ArgumentError("kind must be :bars or :heatmap, got :$kind"))
end

function _factorplot(::Val{:pbp}, f::Factor; kind::Symbol=:bars, figure=NamedTuple(), axis=NamedTuple(),
                     kwargs...)
    n = length(f.vars)
    n == 0 && throw(ArgumentError("this factor is a single number ($(f.values[])), nothing to plot"))
    n <= 3 || throw(ArgumentError("factorplot draws factors of up to three variables; this one has $(n)"))
    if n <= 2
        fig = PBP.Figure(; size=(n == 1 ? 420 : 520, 340), figure...)
        ax = PBP.Axis(fig[1, 1]; title=f.label, axis...)
        factorplot!(ax, f; kind, kwargs...)
        return fig
    end
    v3, d3 = f.vars[3], f.domains[3]
    fig = PBP.Figure(; size=(340 * length(d3), 340), title=f.label, figure...)
    for (k, val) in enumerate(d3)
        slice = Factor(f.vars[1:2], f.domains[1:2], f.values[:, :, k], f.label)
        ax = PBP.Axis(fig[1, k]; title=string(v3.name, " = ", _pvaluelabels(v3, d3)[k]), axis...)
        factorplot!(ax, slice; kind, legend=(k == length(d3)), kwargs...)
        k == 1 || (ax.ylabel = "")
    end
    return fig
end

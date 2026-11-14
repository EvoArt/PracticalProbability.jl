# Plots of Factors (results of `compute`): bars, grouped bars, heatmaps.

# Tick labels for a variable's values: an event's [true, false] as A, Aᶜ.
function _valuelabels(v, dom)
    if PP.isevent(v) && length(dom) == 2 && all(x -> x isa Bool, dom)
        return [d ? string(v.name) : string(v.name, "ᶜ") for d in dom]
    end
    return [string(d) for d in dom]
end

function factorplot!(ax, f::Factor; kind::Symbol=:bars, color=Makie.wong_colors()[1],
                     palette=Makie.wong_colors(), colormap=:viridis, values::Bool=true)
    n = length(f.vars)
    n in (1, 2) || throw(ArgumentError("factorplot! draws factors of one or two variables; " *
                                       "this one has $(n). Use factorplot, or sumout first"))
    v1, d1 = f.vars[1], f.domains[1]
    xs = collect(1:length(d1))
    ax.xticks = (xs, _valuelabels(v1, d1))
    ax.xlabel = string(v1.name)
    if n == 1
        Makie.barplot!(ax, xs, vec(f.values); color)
        ax.ylabel = isempty(f.label) ? "" : f.label
        return ax
    end
    v2, d2 = f.vars[2], f.domains[2]
    if kind === :heatmap
        ys = collect(1:length(d2))
        hm = Makie.heatmap!(ax, xs, ys, f.values; colormap)
        ax.yticks = (ys, _valuelabels(v2, d2))
        ax.ylabel = string(v2.name)
        if values
            lo, hi = extrema(f.values)
            for i in xs, j in ys
                x = f.values[i, j]
                Makie.text!(ax, Makie.Point2f(i, j); text=_fmt(x), align=(:center, :center),
                            color=(hi > lo && (x - lo) / (hi - lo) > 0.6) ? :black : :white)
            end
        end
        return hm
    elseif kind === :bars
        n1, n2 = length(d1), length(d2)
        x = repeat(xs; outer=n2)
        group = repeat(1:n2; inner=n1)
        Makie.barplot!(ax, x, vec(f.values); dodge=group, color=[palette[mod1(g, length(palette))] for g in group])
        ax.ylabel = isempty(f.label) ? "" : f.label
        return ax
    end
    throw(ArgumentError("kind must be :bars or :heatmap, got :$kind"))
end

function _factorlegend!(pos, f::Factor, palette)
    v2, d2 = f.vars[2], f.domains[2]
    elems = [Makie.PolyElement(color=palette[mod1(g, length(palette))]) for g in eachindex(d2)]
    return Makie.Legend(pos, elems, _valuelabels(v2, d2), string(v2.name); framevisible=false)
end

function PP._factorplot(::Val{:makie}, f::Factor; kind::Symbol=:bars, figure=NamedTuple(), axis=NamedTuple(),
                        palette=Makie.wong_colors(), kwargs...)
    n = length(f.vars)
    n == 0 && throw(ArgumentError("this factor is a single number ($(f.values[])), nothing to plot"))
    n <= 3 || throw(ArgumentError("factorplot draws factors of up to three variables; this one has $(n)"))
    title = isempty(f.label) ? "" : f.label
    if n <= 2
        fig = Makie.Figure(; size=(n == 1 ? 420 : 520, 340), figure...)
        ax = Makie.Axis(fig[1, 1]; title, axis...)
        hm = factorplot!(ax, f; kind, palette, kwargs...)
        n == 2 && kind === :bars && _factorlegend!(fig[1, 2], f, palette)
        n == 2 && kind === :heatmap && Makie.Colorbar(fig[1, 2], hm)
        return fig
    end
    v3, d3 = f.vars[3], f.domains[3]
    fig = Makie.Figure(; size=(360 * length(d3) + 120, 340), figure...)
    for (k, val) in enumerate(d3)
        slice = Factor(f.vars[1:2], f.domains[1:2], f.values[:, :, k], f.label)
        ax = Makie.Axis(fig[1, k]; title=string(v3.name, " = ", _valuelabels(v3, d3)[k]), axis...)
        factorplot!(ax, slice; kind, palette, kwargs...)
        k == 1 || (ax.ylabel = "")
    end
    kind === :bars && _factorlegend!(fig[1, length(d3)+1], f, palette)
    isempty(title) || Makie.Label(fig[0, :], title; font=:bold)
    return fig
end

# barplot(f) and heatmap(f) for one- and two-variable factors
function Makie.convert_arguments(P::Type{<:Makie.BarPlot}, f::Factor)
    length(f.vars) == 1 || throw(ArgumentError("barplot(f) needs a factor of one variable; use factorplot"))
    return Makie.convert_arguments(P, collect(1:length(f.domains[1])), vec(f.values))
end
function Makie.convert_arguments(P::Type{<:Makie.Heatmap}, f::Factor)
    length(f.vars) == 2 || throw(ArgumentError("heatmap(f) needs a factor of two variables; use factorplot"))
    return Makie.convert_arguments(P, collect(1:length(f.domains[1])), collect(1:length(f.domains[2])), f.values)
end

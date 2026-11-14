# DAG views (layouts: dagplot.jl) with PracticalBayesPlot.

const _PDAGFONT = (node=0.24, title=0.2, caption=0.22, open=0.15, blocked=0.15, note=0.15, plate=0.15)

function _pdagstyle(; highlightcolor=1, opencolor=:firebrick, blockedcolor=:gray50, blockercolor=:darkorange,
                    observedcolor=:gray82, strokecolor=:black, textcolor=:black, strokewidth=1.6, textscale=1.0,
                    labels=:latex)
    return (; highlight=_rgb3(highlightcolor), open=_rgb3(opencolor), blocked=_rgb3(blockedcolor),
            blocker=_rgb3(blockercolor), observed=_pwhite(observedcolor, 1.0), stroke=_rgb3(strokecolor),
            text=_rgb3(textcolor), width=Float64(strokewidth), scale=Float64(textscale), labels)
end

_pview(x::DAGView, q, plates, adjust) = x
_pview(x::DAGView, ::Nothing, plates, adjust) = x

_pview(x, ::Nothing, plates, adjust) = dag_view(x; plates, adjust)
_pview(x, q, plates, adjust) = dag_view(x, q; plates, adjust)

function dagplot!(ax::PBP.Axis, x, q=nothing; plates=nothing, adjust=nothing, kwargs...)
    V = _pview(x, q, plates, adjust)
    _pdraw_dag!(ax, V, _pdagstyle(; kwargs...))
    return V
end

function _dagplot(::Val{:pbp}, x, q=nothing; plates=nothing, adjust=nothing, scale::Real=110,
                  figure=NamedTuple(), axis=NamedTuple(), kwargs...)
    V = _pview(x, q, plates, adjust)
    fig, ax = _pfigure(V.bbox, scale, figure, axis)
    _pdraw_dag!(ax, V, _pdagstyle(; kwargs...))
    return fig
end

# points along the quadratic Bézier curve from p0 to p1 with control point c
_pqbez(p0, c, p1; n=32) = [((1 - t)^2 * p0[1] + 2(1 - t) * t * c[1] + t^2 * p1[1],
                            (1 - t)^2 * p0[2] + 2(1 - t) * t * c[2] + t^2 * p1[2]) for t in range(0, 1; length=n)]

function _pdraw_dag!(ax, V::DAGView, st)
    R = DAG_RADIUS
    for panel in V.panels
        pos = Dict(n.v => n for n in panel.nodes)
        for q in panel.plates
            PBP.lines!(ax, [q.x0, q.x1, q.x1, q.x0, q.x0], [q.y0, q.y0, q.y1, q.y1, q.y0];
                       color=st.stroke, linewidth=0.8st.width)
            _ptext!(ax, q.x1 - 0.07, q.y0 + 0.06, q.label; align=(:right, :bottom),
                    fontsize=_PDAGFONT.plate * st.scale, color=st.text)
        end
        for e in panel.edges
            a, b = pos[e.from], pos[e.to]
            ra = a.shape === :dot ? 0.09 : R
            rb = b.shape === :dot ? 0.09 : R
            hypot(b.x - a.x, b.y - a.y) > ra + rb || continue
            c = e.bend
            # leave a toward the control point (or b), arrive at b from it
            from = c === nothing ? (b.x, b.y) : c
            to = c === nothing ? (a.x, a.y) : c
            u0 = (from[1] - a.x, from[2] - a.y) ./ hypot(from[1] - a.x, from[2] - a.y)
            u1 = (b.x - to[1], b.y - to[2]) ./ hypot(b.x - to[1], b.y - to[2])
            p0 = (a.x + u0[1] * ra, a.y + u0[2] * ra)
            p1 = (b.x - u1[1] * rb, b.y - u1[2] * rb)
            col, lw, ls = e.style === :open ? (st.open, 2.2st.width, :solid) :
                          e.style === :blocked ? (st.blocked, st.width, :dash) :
                          e.style === :removed ? (_over(st.stroke, 0.25, _WHITE), st.width, :dot) :
                          (st.stroke, st.width, :solid)
            head, hw = 0.17, 0.075
            base = (p1[1] - u1[1] * head, p1[2] - u1[2] * head)
            pts = c === nothing ? [p0, base] : _pqbez(p0, c, base)
            PBP.lines!(ax, first.(pts), last.(pts); color=col, linewidth=lw, linestyle=ls)
            PBP.poly!(ax, [p1, (base[1] - u1[2] * hw, base[2] + u1[1] * hw), (base[1] + u1[2] * hw, base[2] - u1[1] * hw)];
                      color=col)
        end
        for n in panel.nodes
            fill = n.fill === :observed ? st.observed :
                   n.fill === :highlight ? _over(st.highlight, 0.35, _WHITE) : _WHITE
            lab = _plabel(n.v, st.labels === :none ? :latex : st.labels)
            fs = _PDAGFONT.node * min(1.0, 3.2 / max(length(string(n.v.name)), 1)) * st.scale
            if n.shape === :dot
                PBP.poly!(ax, _pcircle(n.x, n.y, 0.09; n=40); color=st.stroke)
                # the name above the dot: an input is a root, so its edges leave downwards
                _ptext!(ax, n.x, n.y + 0.13, lab; align=(:center, :bottom), fontsize=fs, color=st.text)
                continue
            end
            h = 0.92R
            pts = n.shape === :square ? [(n.x - h, n.y - h), (n.x + h, n.y - h), (n.x + h, n.y + h), (n.x - h, n.y + h)] :
                  _pcircle(n.x, n.y, R)
            PBP.poly!(ax, pts; color=fill, strokecolor=n.endpoint ? st.highlight : st.stroke,
                      strokewidth=n.endpoint ? 2.4st.width : st.width)
            if n.blocks
                c = _pcircle(n.x, n.y, R + 0.075)
                PBP.lines!(ax, [first.(c); c[1][1]], [last.(c); c[1][2]]; color=st.blocker, linewidth=1.6st.width,
                           linestyle=:dash)
            end
            _ptext!(ax, n.x, n.y, lab; fontsize=fs, color=st.text)
        end
    end
    for t in V.texts
        col = t.role === :open ? st.open : t.role === :blocked ? st.blocked : st.text
        _ptext!(ax, t.x, t.y, (t.text, something(t.latex, "")); align=(t.halign, :center),
                fontsize=getfield(_PDAGFONT, t.role) * st.scale, color=col)
    end
    return ax
end

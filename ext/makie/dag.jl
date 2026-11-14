# DAG drawings (layouts: src/dagplot.jl).

const DAGFONT = (node=0.24, title=0.2, caption=0.22, open=0.15, blocked=0.15, note=0.15, plate=0.15)

function _dagstyle(; highlightcolor=Makie.wong_colors()[1], opencolor=:firebrick, blockedcolor=:gray50,
                   blockercolor=:darkorange, observedcolor=:gray82, strokecolor=:black, textcolor=:black,
                   strokewidth=1.6, textscale=1.0, labels=:latex)
    return (; highlight=_rgba(highlightcolor), open=_rgba(opencolor), blocked=_rgba(blockedcolor),
            blocker=_rgba(blockercolor), observed=_rgba(observedcolor), stroke=_rgba(strokecolor),
            text=_rgba(textcolor), width=strokewidth, scale=textscale, labels)
end

_view(x::DAGView, q, plates, adjust) = x
_view(x::DAGView, ::Nothing, plates, adjust) = x
_view(x, ::Nothing, plates, adjust) = dag_view(x; plates, adjust)
_view(x, q, plates, adjust) = dag_view(x, q; plates, adjust)

function dagplot!(ax, x, q=nothing; plates=nothing, adjust=nothing, kwargs...)
    V = _view(x, q, plates, adjust)
    _draw_dag!(ax, V, _dagstyle(; kwargs...))
    return V
end

function PP._dagplot(::Val{:makie}, x, q=nothing; plates=nothing, adjust=nothing, scale::Real=110,
                     figure=NamedTuple(), axis=NamedTuple(), kwargs...)
    V = _view(x, q, plates, adjust)
    fig, ax = _figure(V.bbox, scale, figure, axis)
    _draw_dag!(ax, V, _dagstyle(; kwargs...))
    return fig
end

_square(x, y, h) = Makie.Point2f[(x - h, y - h), (x + h, y - h), (x + h, y + h), (x - h, y + h)]

function _nodelabel(v, mode)
    lab = _label(v, mode === :none ? :latex : mode)
    # visible length: `theta` is the single glyph θ in LaTeX
    vis = mode === :text ? string(v.name) : replace(PP._lx(v.name), r"\\[a-zA-Z]+" => "x", r"[{}_^\s]" => "")
    return lab, DAGFONT.node * min(1.0, 3.2 / max(length(vis), 1))
end

_qbez(p0, c, p1; n=32) = [Makie.Point2f((1 - t)^2 * p0[1] + 2(1 - t) * t * c[1] + t^2 * p1[1],
                                        (1 - t)^2 * p0[2] + 2(1 - t) * t * c[2] + t^2 * p1[2]) for t in range(0, 1; length=n)]

function _draw_dag!(ax, V::DAGView, st)
    R = PP.DAG_RADIUS
    white = Makie.RGBAf(1, 1, 1, 1)
    for panel in V.panels
        pos = Dict(n.v => n for n in panel.nodes)
        for q in panel.plates
            Makie.lines!(ax, Makie.Point2f[(q.x0, q.y0), (q.x1, q.y0), (q.x1, q.y1), (q.x0, q.y1), (q.x0, q.y0)];
                         color=st.stroke, linewidth=st.width * 0.8)
            Makie.text!(ax, Makie.Point2f(q.x1 - 0.07, q.y0 + 0.06); text=q.label, align=(:right, :bottom),
                        fontsize=DAGFONT.plate * st.scale, markerspace=:data, color=st.text)
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
            col, lw, ls = e.style === :open ? (st.open, 2.2st.width, nothing) :
                          e.style === :blocked ? (st.blocked, st.width, :dash) :
                          e.style === :removed ? (_onwhite(st.stroke, 0.25), st.width, :dot) :
                          (st.stroke, st.width, nothing)
            head, hw = 0.17, 0.075
            base = (p1[1] - u1[1] * head, p1[2] - u1[2] * head)
            pts = c === nothing ? Makie.Point2f[p0, base] : _qbez(p0, c, base)
            Makie.lines!(ax, pts; color=col, linewidth=lw, linestyle=ls)
            Makie.poly!(ax, Makie.Point2f[p1, (base[1] - u1[2] * hw, base[2] + u1[1] * hw),
                                          (base[1] + u1[2] * hw, base[2] - u1[1] * hw)];
                        color=col, strokewidth=0)
        end
        for n in panel.nodes
            fill = n.fill === :observed ? _onwhite(st.observed, 1.0) :
                   n.fill === :highlight ? _onwhite(st.highlight, 0.35) : white
            lab, fs = _nodelabel(n.v, st.labels)
            if n.shape === :dot
                Makie.poly!(ax, _circle(n.x, n.y, 0.09; n=40); color=st.stroke, strokewidth=0)
                # the name above the dot: an input is a root, so its edges leave downwards
                Makie.text!(ax, Makie.Point2f(n.x, n.y + 0.13); text=lab, align=(:center, :bottom),
                            fontsize=fs * st.scale, markerspace=:data, color=st.text)
                continue
            end
            pts = n.shape === :square ? _square(n.x, n.y, 0.92R) : _circle(n.x, n.y, R; n=72)
            Makie.poly!(ax, pts; color=fill, strokecolor=n.endpoint ? st.highlight : st.stroke,
                        strokewidth=n.endpoint ? 2.4st.width : st.width)
            n.blocks && Makie.lines!(ax, _circle(n.x, n.y, R + 0.075; n=72); color=st.blocker,
                                     linewidth=1.6st.width, linestyle=:dash)
            Makie.text!(ax, Makie.Point2f(n.x, n.y); text=lab, align=(:center, :center),
                        fontsize=fs * st.scale, markerspace=:data, color=st.text)
        end
    end
    for t in V.texts
        col = t.role === :open ? st.open : t.role === :blocked ? st.blocked : st.text
        txt = t.latex === nothing ? t.text : _tex(t.latex, t.text)
        Makie.text!(ax, Makie.Point2f(t.x, t.y); text=txt, align=(t.halign, :center),
                    fontsize=getfield(DAGFONT, t.role) * st.scale, markerspace=:data, color=col)
    end
    return ax
end

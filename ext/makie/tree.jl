# Probability trees and mosaic plots (layouts: src/probtree.jl). Leaves and
# tiles are coloured like Venn panels: the event measured (E ∩ C) filled, the
# rest of the condition C tinted, outside C grey.

function _treestyle(; highlightcolor=Makie.wong_colors()[1], excludedcolor=:gray88, strokecolor=:black,
                    textcolor=:black, strokewidth=1.4, textscale=1.0, labels=:latex, palette=Makie.wong_colors())
    hl = _rgba(highlightcolor)
    return (; highlight=_onwhite(hl, 1.0), given=_onwhite(hl, 0.2), excluded=_onwhite(excludedcolor, 1.0),
            stroke=_rgba(strokecolor), text=_rgba(textcolor), width=strokewidth, scale=textscale, labels,
            palette=[_onwhite(c, 0.3) for c in palette])
end

_rolecolour(st, role) = role === :target ? st.highlight : role === :given ? st.given :
                        role === :excluded ? st.excluded : Makie.RGBAf(1, 1, 1, 1)
_roletext(st, role) = role === :target ? Makie.RGBAf(1, 1, 1, 1) : st.text

_caption(latex, plain) = latex === nothing ? plain : _tex(latex, plain)

# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------

const TREEFONT = PP.TREE_FONT

_treelayout(x; order, bn, facts, samples, dag) =
    x isa TreeLayout ? x : tree_layout(x; order, bn, facts, samples, dag)

function probtree!(ax, x; order=nothing, bn=nothing, facts=nothing, samples=nothing, dag=nothing, kwargs...)
    L = _treelayout(x; order, bn, facts, samples, dag)
    _draw_tree!(ax, L, _treestyle(; kwargs...))
    return L
end

function PP._probtree(::Val{:makie}, x; order=nothing, bn=nothing, facts=nothing, samples=nothing, dag=nothing,
                      scale::Real=120, figure=NamedTuple(), axis=NamedTuple(), kwargs...)
    L = _treelayout(x; order, bn, facts, samples, dag)
    fig, ax = _figure(L.bbox, scale, figure, axis)
    _draw_tree!(ax, L, _treestyle(; kwargs...))
    return fig
end

function _draw_tree!(ax, L::TreeLayout, st)
    mid = PP._branchmiddle(L)
    fo, fp = TREEFONT.outcome * st.scale, TREEFONT.prob * st.scale
    mode = st.labels === :none ? :latex : st.labels
    for (b, m) in zip(L.branches, mid)
        Makie.lines!(ax, Makie.Point2f[(b.x0, b.y0), (b.x1, b.y1)]; color=st.stroke, linewidth=st.width)
        # the outcome by the branch's end, P(outcome | path) = value along it, both
        # on the outer side and clear of the line (see PP._branchlabels)
        suffix = b.value === nothing ? "" : " = " * _fmt(b.value)
        plain = st.labels === :none ? (b.value === nothing ? "" : _fmt(b.value)) : sprint(show, b.prob) * suffix
        lab = st.labels === :none ? plain : _label(b.prob, st.labels; suffix)
        (px, py), (ox, oy) = PP._branchlabels(b, PP._textwidth(plain, fp), fp,
                                              PP._textwidth(sprint(show, b.atom), fo), fo; middle=m)
        Makie.text!(ax, Makie.Point2f(ox, oy); text=_label(b.atom, mode), align=(:center, :center),
                    fontsize=fo, markerspace=:data, color=st.text)
        isempty(plain) || Makie.text!(ax, Makie.Point2f(px, py); text=lab, align=(:center, :center), fontsize=fp,
                                      markerspace=:data, color=st.text)
        Makie.scatter!(ax, Makie.Point2f(b.x0, b.y0); color=st.stroke, markersize=6)
    end
    for l in L.leaves
        joint = PP._mkprob(l.path, PP.Atom[])
        lab = _label(joint, mode; suffix=l.joint === nothing ? "" : " = " * _fmt(l.joint))
        x0, w = PP._leafbox(l, st.scale)
        Makie.poly!(ax, Makie.Point2f[(x0, l.y - 0.22), (x0 + w, l.y - 0.22), (x0 + w, l.y + 0.22), (x0, l.y + 0.22)];
                    color=_rolecolour(st, l.role), strokecolor=_onwhite(st.stroke, 0.35), strokewidth=0.8)
        Makie.text!(ax, Makie.Point2f(x0 + 0.1, l.y); text=lab, align=(:left, :center),
                    fontsize=TREEFONT.leaf * st.scale, markerspace=:data, color=_roletext(st, l.role))
        Makie.scatter!(ax, Makie.Point2f(l.x, l.y); color=st.stroke, markersize=6)
    end
    ymin = minimum(l.y for l in L.leaves)
    y = ymin - 0.65
    if !isempty(L.caption)
        Makie.text!(ax, Makie.Point2f(0.0, y); text=_caption(L.captionlatex, L.caption), align=(:left, :center),
                    fontsize=TREEFONT.caption * st.scale, markerspace=:data, color=st.text)
        y -= 0.42
    end
    isempty(L.legend) || Makie.text!(ax, Makie.Point2f(0.0, y); text=L.legend, align=(:left, :center),
                                     fontsize=TREEFONT.legend * st.scale, markerspace=:data, color=st.text)
    return ax
end

# ---------------------------------------------------------------------------
# Mosaics
# ---------------------------------------------------------------------------

const MOSAICFONT = PP.MOSAIC_FONT

_mosaiclayout(x; order, bn, facts, samples) =
    x isa MosaicLayout ? x : mosaic_layout(x; order, bn, facts, samples)

function mosaic!(ax, x; order=nothing, bn=nothing, facts=nothing, samples=nothing, kwargs...)
    L = _mosaiclayout(x; order, bn, facts, samples)
    _draw_mosaic!(ax, L, _treestyle(; kwargs...))
    return L
end

_mosaicbelow(L::MosaicLayout) = (isempty(L.caption) ? 0.0 : 0.13) + (isempty(L.legend) ? 0.0 : 0.07)

function PP._mosaic(::Val{:makie}, x; order=nothing, bn=nothing, facts=nothing, samples=nothing, scale::Real=520,
                    figure=NamedTuple(), axis=NamedTuple(), kwargs...)
    L = _mosaiclayout(x; order, bn, facts, samples)
    st = _treestyle(; kwargs...)
    _, _, bbox = PP._mosaicplaces(L, _mosaicbelow(L), st.scale)
    fig, ax = _figure(bbox, scale, figure, axis; m=0.03)
    _draw_mosaic!(ax, L, st)
    return fig
end

function _draw_mosaic!(ax, L::MosaicLayout, st)
    first_outcomes = [h[1] for h in L.headers]
    mode = st.labels === :none ? :latex : st.labels
    for (t, (x0, x1, y0, y1)) in zip(L.tiles, L.rects)
        col = if t.role === :none
            k = something(findfirst(==(t.path[1]), first_outcomes), 1)
            st.palette[mod1(k, length(st.palette))]
        else
            _rolecolour(st, t.role)
        end
        Makie.poly!(ax, Makie.Point2f[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]; color=col,
                    strokecolor=_onwhite(st.stroke, 0.5), strokewidth=0.8)
        # label the tile if the label fits, shrunk a little if need be
        ev = PP._evintersect(t.path)
        fs = PP._tilefont(sprint(show, ev), x1 - x0, y1 - y0, MOSAICFONT.tile * st.scale)
        fs > 0 || continue
        tc = _roletext(st, t.role)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        Makie.text!(ax, Makie.Point2f(cx, cy + 0.45fs); text=_label(ev, mode), align=(:center, :bottom),
                    fontsize=fs, markerspace=:data, color=tc)
        t.joint === nothing || Makie.text!(ax, Makie.Point2f(cx, cy - 0.2fs); text=_fmt(t.joint),
                                           align=(:center, :top), fontsize=fs, markerspace=:data, color=tc)
    end
    hx, sy, _ = PP._mosaicplaces(L, _mosaicbelow(L), st.scale)
    fh = MOSAICFONT.header * st.scale
    for ((a, _), x) in zip(L.headers, hx)
        Makie.text!(ax, Makie.Point2f(x, 1.035); text=_label(a, mode),
                    align=(:center, :center), fontsize=fh, markerspace=:data, color=st.text)
    end
    for ((a, _), y) in zip(L.sideheaders, sy)
        Makie.text!(ax, Makie.Point2f(1.02, y); text=_label(a, mode),
                    align=(:left, :center), fontsize=fh, markerspace=:data, color=st.text)
    end
    y = -0.075
    if !isempty(L.caption)
        Makie.text!(ax, Makie.Point2f(0.5, y); text=_caption(L.captionlatex, L.caption), align=(:center, :center),
                    fontsize=MOSAICFONT.caption * st.scale, markerspace=:data, color=st.text)
        y -= 0.09
    end
    isempty(L.legend) || Makie.text!(ax, Makie.Point2f(0.5, y); text=L.legend, align=(:center, :center),
                                     fontsize=MOSAICFONT.legend * st.scale, markerspace=:data, color=st.text)
    return ax
end

# Probability trees and mosaics (layouts: probtree.jl) with PracticalBayesPlot.
# Leaves and tiles are coloured like Venn panels: the event measured (E ∩ C)
# filled, the rest of the condition C tinted, outside C grey.

function _ptreestyle(; highlightcolor=1, excludedcolor=:gray88, strokecolor=:black, textcolor=:black,
                     strokewidth=1.4, textscale=1.0, labels=:latex, palette=PBP.wong_colors())
    return (; highlight=_pwhite(highlightcolor, 1.0), given=_pwhite(highlightcolor, 0.2),
            excluded=_pwhite(excludedcolor, 1.0), stroke=_rgb3(strokecolor), text=_rgb3(textcolor),
            width=Float64(strokewidth), scale=Float64(textscale), labels,
            palette=[_pwhite(c, 0.3) for c in palette])
end

_prolecolour(st, role) = role === :target ? st.highlight : role === :given ? st.given :
                         role === :excluded ? st.excluded : _WHITE
_proletext(st, role) = role === :target ? _WHITE : st.text
_plabelmode(st) = st.labels === :none ? :latex : st.labels
_pcaption(latex, plain) = (plain, something(latex, ""))

# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------

const _PTREEFONT = TREE_FONT

_ptreelayout(x; order, bn, facts, samples, dag) =
    x isa TreeLayout ? x : tree_layout(x; order, bn, facts, samples, dag)

function probtree!(ax::PBP.Axis, x; order=nothing, bn=nothing, facts=nothing, samples=nothing, dag=nothing,
                   kwargs...)
    L = _ptreelayout(x; order, bn, facts, samples, dag)
    _pdraw_tree!(ax, L, _ptreestyle(; kwargs...))
    return L
end

function _probtree(::Val{:pbp}, x; order=nothing, bn=nothing, facts=nothing, samples=nothing, dag=nothing,
                   scale::Real=120, figure=NamedTuple(), axis=NamedTuple(), kwargs...)
    L = _ptreelayout(x; order, bn, facts, samples, dag)
    fig, ax = _pfigure(L.bbox, scale, figure, axis)
    _pdraw_tree!(ax, L, _ptreestyle(; kwargs...))
    return fig
end

function _pdraw_tree!(ax, L::TreeLayout, st)
    mid = _branchmiddle(L)
    fo, fp = _PTREEFONT.outcome * st.scale, _PTREEFONT.prob * st.scale
    for (b, m) in zip(L.branches, mid)
        PBP.lines!(ax, [b.x0, b.x1], [b.y0, b.y1]; color=st.stroke, linewidth=st.width)
        # the outcome by the branch's end, P(outcome | path) = value along it (see _branchlabels)
        out = _plabel(b.atom, _plabelmode(st))
        lab = st.labels === :none ? (b.value === nothing ? ("", "") : (_plainnum(b.value), "")) :
              _plabel(b.prob, st.labels; suffix=b.value === nothing ? "" : " = " * _plainnum(b.value))
        (px, py), (ox, oy) = _branchlabels(b, _textwidth(lab[1], fp), fp, _textwidth(out[1], fo), fo; middle=m)
        _ptext!(ax, ox, oy, out; fontsize=fo, color=st.text)
        _ptext!(ax, px, py, lab; fontsize=fp, color=st.text)
        PBP.scatter!(ax, [b.x0], [b.y0]; color=st.stroke, markersize=6)
    end
    for l in L.leaves
        lab = _plabel(_mkprob(l.path, Atom[]), _plabelmode(st); suffix=l.joint === nothing ? "" : " = " * _plainnum(l.joint))
        x0, w = _leafbox(l, st.scale)
        PBP.poly!(ax, [(x0, l.y - 0.22), (x0 + w, l.y - 0.22), (x0 + w, l.y + 0.22), (x0, l.y + 0.22)];
                  color=_prolecolour(st, l.role), strokecolor=_over(st.stroke, 0.35, _WHITE), strokewidth=0.8)
        _ptext!(ax, x0 + 0.1, l.y, lab; align=(:left, :center), fontsize=_PTREEFONT.leaf * st.scale,
                color=_proletext(st, l.role))
        PBP.scatter!(ax, [l.x], [l.y]; color=st.stroke, markersize=6)
    end
    y = minimum(l.y for l in L.leaves) - 0.65
    if !isempty(L.caption)
        _ptext!(ax, 0.0, y, _pcaption(L.captionlatex, L.caption); align=(:left, :center),
                fontsize=_PTREEFONT.caption * st.scale, color=st.text)
        y -= 0.42
    end
    isempty(L.legend) || _ptext!(ax, 0.0, y, L.legend; align=(:left, :center),
                                 fontsize=_PTREEFONT.legend * st.scale, color=st.text)
    return ax
end

# ---------------------------------------------------------------------------
# Mosaics
# ---------------------------------------------------------------------------

const _PMOSAICFONT = MOSAIC_FONT

_pmosaiclayout(x; order, bn, facts, samples) =
    x isa MosaicLayout ? x : mosaic_layout(x; order, bn, facts, samples)

function mosaic!(ax::PBP.Axis, x; order=nothing, bn=nothing, facts=nothing, samples=nothing, kwargs...)
    L = _pmosaiclayout(x; order, bn, facts, samples)
    _pdraw_mosaic!(ax, L, _ptreestyle(; kwargs...))
    return L
end

_pmosaicbelow(L::MosaicLayout) = (isempty(L.caption) ? 0.0 : 0.13) + (isempty(L.legend) ? 0.0 : 0.07)

function _mosaic(::Val{:pbp}, x; order=nothing, bn=nothing, facts=nothing, samples=nothing, scale::Real=520,
                 figure=NamedTuple(), axis=NamedTuple(), kwargs...)
    L = _pmosaiclayout(x; order, bn, facts, samples)
    st = _ptreestyle(; kwargs...)
    _, _, bbox = _mosaicplaces(L, _pmosaicbelow(L), st.scale)
    fig, ax = _pfigure(bbox, scale, figure, axis; m=0.03)
    _pdraw_mosaic!(ax, L, st)
    return fig
end

function _pdraw_mosaic!(ax, L::MosaicLayout, st)
    first_outcomes = [h[1] for h in L.headers]
    mode = _plabelmode(st)
    for (t, (x0, x1, y0, y1)) in zip(L.tiles, L.rects)
        col = if t.role === :none
            k = something(findfirst(==(t.path[1]), first_outcomes), 1)
            st.palette[mod1(k, length(st.palette))]
        else
            _prolecolour(st, t.role)
        end
        PBP.poly!(ax, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]; color=col,
                  strokecolor=_over(st.stroke, 0.5, _WHITE), strokewidth=0.8)
        # label the tile if the label fits, shrunk a little if need be
        lab = _plabel(_evintersect(t.path), mode)
        fs = _tilefont(lab[1], x1 - x0, y1 - y0, _PMOSAICFONT.tile * st.scale)
        fs > 0 || continue
        tc = _proletext(st, t.role)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        _ptext!(ax, cx, cy + 0.45fs, lab; align=(:center, :bottom), fontsize=fs, color=tc)
        t.joint === nothing || _ptext!(ax, cx, cy - 0.2fs, _plainnum(t.joint); align=(:center, :top), fontsize=fs,
                                       color=tc)
    end
    hx, sy, _ = _mosaicplaces(L, _pmosaicbelow(L), st.scale)
    fh = _PMOSAICFONT.header * st.scale
    for ((a, _), x) in zip(L.headers, hx)
        _ptext!(ax, x, 1.035, _plabel(a, mode); fontsize=fh, color=st.text)
    end
    for ((a, _), y) in zip(L.sideheaders, sy)
        _ptext!(ax, 1.02, y, _plabel(a, mode); align=(:left, :center), fontsize=fh, color=st.text)
    end
    y = -0.075
    if !isempty(L.caption)
        _ptext!(ax, 0.5, y, _pcaption(L.captionlatex, L.caption); fontsize=_PMOSAICFONT.caption * st.scale,
                color=st.text)
        y -= 0.09
    end
    isempty(L.legend) || _ptext!(ax, 0.5, y, L.legend; fontsize=_PMOSAICFONT.legend * st.scale, color=st.text)
    return ax
end

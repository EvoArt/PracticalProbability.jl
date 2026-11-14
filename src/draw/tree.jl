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

const _PTREEFONT = (outcome=0.2, prob=0.16, leaf=0.17, caption=0.2, legend=0.16)

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
    for b in L.branches
        PBP.lines!(ax, [b.x0, b.x1], [b.y0, b.y1]; color=st.stroke, linewidth=st.width)
        mx, my = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
        # the outcome at the branch's end, its conditional probability and value along the branch
        _ptext!(ax, b.x1 - 0.12, b.y1 + 0.05, _plabel(b.atom, _plabelmode(st)); align=(:right, :bottom),
                fontsize=_PTREEFONT.outcome * st.scale, color=st.text)
        st.labels === :none || _ptext!(ax, mx, my + 0.06, _plabel(b.prob, st.labels); align=(:center, :bottom),
                                       fontsize=_PTREEFONT.prob * st.scale, color=st.text)
        b.value === nothing || _ptext!(ax, mx, my - 0.06, _plainnum(b.value); align=(:center, :top),
                                       fontsize=_PTREEFONT.prob * st.scale, color=st.text)
        PBP.scatter!(ax, [b.x0], [b.y0]; color=st.stroke, markersize=6)
    end
    for l in L.leaves
        joint = _mkprob(l.path, Atom[])
        lab = _plabel(joint, _plabelmode(st); suffix=l.joint === nothing ? "" : " = " * _plainnum(l.joint))
        w = 0.11 * (length(sprint(show, joint)) + (l.joint === nothing ? 0 : 8))
        x0 = l.x + 0.15
        PBP.poly!(ax, [(x0, l.y - 0.22), (x0 + w, l.y - 0.22), (x0 + w, l.y + 0.22), (x0, l.y + 0.22)];
                  color=_prolecolour(st, l.role), strokecolor=_over(st.stroke, 0.35, _WHITE), strokewidth=0.8)
        _ptext!(ax, x0 + 0.08, l.y, lab; align=(:left, :center), fontsize=_PTREEFONT.leaf * st.scale,
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

const _PMOSAICFONT = (tile=0.04, header=0.045, caption=0.042, legend=0.034)

_pmosaiclayout(x; order, bn, facts, samples) =
    x isa MosaicLayout ? x : mosaic_layout(x; order, bn, facts, samples)

function mosaic!(ax::PBP.Axis, x; order=nothing, bn=nothing, facts=nothing, samples=nothing, kwargs...)
    L = _pmosaiclayout(x; order, bn, facts, samples)
    _pdraw_mosaic!(ax, L, _ptreestyle(; kwargs...))
    return L
end

function _mosaic(::Val{:pbp}, x; order=nothing, bn=nothing, facts=nothing, samples=nothing, scale::Real=520,
                 figure=NamedTuple(), axis=NamedTuple(), kwargs...)
    L = _pmosaiclayout(x; order, bn, facts, samples)
    below = (isempty(L.caption) ? 0.0 : 0.09) + (isempty(L.legend) ? 0.0 : 0.07)
    fig, ax = _pfigure((0.0, 1.0, -0.02 - below, 1.08), scale, figure, axis; m=0.03)
    _pdraw_mosaic!(ax, L, _ptreestyle(; kwargs...))
    return fig
end

function _pdraw_mosaic!(ax, L::MosaicLayout, st)
    first_outcomes = [h[1] for h in L.headers]
    for (t, (x0, x1, y0, y1)) in zip(L.tiles, L.rects)
        col = if t.role === :none
            k = something(findfirst(==(t.path[1]), first_outcomes), 1)
            st.palette[mod1(k, length(st.palette))]
        else
            _prolecolour(st, t.role)
        end
        PBP.poly!(ax, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]; color=col,
                  strokecolor=_over(st.stroke, 0.5, _WHITE), strokewidth=0.8)
        # label the tile if it is big enough
        (x1 - x0) > 0.1 && (y1 - y0) > 0.07 || continue
        tc = _proletext(st, t.role)
        _ptext!(ax, (x0 + x1) / 2, (y0 + y1) / 2 + 0.018, _plabel(_evintersect(t.path), _plabelmode(st));
                align=(:center, :bottom), fontsize=_PMOSAICFONT.tile * st.scale, color=tc)
        t.joint === nothing || _ptext!(ax, (x0 + x1) / 2, (y0 + y1) / 2 - 0.008, _plainnum(t.joint);
                                       align=(:center, :top), fontsize=_PMOSAICFONT.tile * st.scale, color=tc)
    end
    for (a, x) in L.headers
        _ptext!(ax, x, 1.035, _plabel(a, _plabelmode(st)); fontsize=_PMOSAICFONT.header * st.scale, color=st.text)
    end
    y = -0.05
    if !isempty(L.caption)
        _ptext!(ax, 0.5, y, _pcaption(L.captionlatex, L.caption); fontsize=_PMOSAICFONT.caption * st.scale,
                color=st.text)
        y -= 0.07
    end
    isempty(L.legend) || _ptext!(ax, 0.5, y, L.legend; fontsize=_PMOSAICFONT.legend * st.scale, color=st.text)
    return ax
end

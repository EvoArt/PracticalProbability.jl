# Venn panels (layouts: venn.jl) with PracticalBayesPlot.
#
# Each cell is filled with an exact polygon. The cell inside circles S (and
# outside the others) is painted as the intersection of the circles in S,
# which is convex; painting cells in order of |S| leaves every point the
# colour of the cell it is really in, since a point inside circles T is
# painted last by T itself. Colours are composited onto the background, so
# every fill is opaque and nothing shows through from below.

const _PVENNFONT = (op=0.16, coef=0.14, note=0.1, label=0.075, set=0.068, cell=0.058)

const _Pt = NTuple{2,Float64}

_lerp(a::_Pt, b::_Pt, t::Float64) = (a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2]))

# the part of the convex polygon P where the affine function f is ≥ 0
function _keep(P::Vector{_Pt}, f)
    out = _Pt[]
    isempty(P) && return out
    prev = P[end]
    fp = f(prev)
    for cur in P
        fc = f(cur)
        if fc >= 0
            fp < 0 && push!(out, _lerp(prev, cur, fp / (fp - fc)))
            push!(out, cur)
        elseif fp >= 0
            push!(out, _lerp(prev, cur, fp / (fp - fc)))
        end
        prev, fp = cur, fc
    end
    return out
end

# P ∩ Q for convex P and convex, counter-clockwise Q
function _intersect_convex(P::Vector{_Pt}, Q::Vector{_Pt})
    n = length(Q)
    for i in 1:n
        a, b = Q[i], Q[mod1(i + 1, n)]
        P = _keep(P, p -> (b[1] - a[1]) * (p[2] - a[2]) - (b[2] - a[2]) * (p[1] - a[1]))
        length(P) < 3 && return _Pt[]
    end
    return P
end

function _pvennstyle(; highlightcolor=1, givencolor=nothing, excludedcolor=(:gray55, 0.4),
                     impossiblecolor=(:gray40, 0.45), strokecolor=:black, strokewidth=1.5, textcolor=:black,
                     labels=:latex, setlabels=true, textscale=1.0, backgroundcolor=:white, pixels=320,
                     cellvalues=:target, cellvaluecolor=:white)
    cellvalues in (:target, :all, :none) ||
        throw(ArgumentError("cellvalues must be :target, :all or :none, got :$cellvalues"))
    labels in (:latex, :text, :none) || throw(ArgumentError("labels must be :latex, :text or :none, got :$labels"))
    bg = _rgb3(backgroundcolor)
    on(c) = (r = PBP.rgba(c); _over((r[1], r[2], r[3]), r[4], bg))
    hl = PBP.rgba(highlightcolor)
    return (; background=bg, highlight=on(highlightcolor),
            given=givencolor === nothing ? _over((hl[1], hl[2], hl[3]), 0.18, bg) : on(givencolor),
            excluded=on(excludedcolor), impossible=on(impossiblecolor), stroke=_rgb3(strokecolor),
            width=Float64(strokewidth), text=_rgb3(textcolor), labels, setlabels, scale=Float64(textscale),
            cellvalues, cellcolor=_rgb3(cellvaluecolor))
end

_pvennlayout(x, sets, facts, samples) = x isa VennLayout ? x : venn_layout(x; sets, facts, samples)

function vennplot!(ax::PBP.Axis, x; sets=nothing, facts=nothing, samples=nothing, kwargs...)
    L = _pvennlayout(x, sets, facts, samples)
    st = _pvennstyle(; kwargs...)
    for panel in L.panels
        _pdraw_vennpanel!(ax, L, panel, st)
    end
    for l in L.lines
        PBP.lines!(ax, [l.x0, l.x1], [l.y0, l.y1]; color=st.stroke, linewidth=st.width)
    end
    for t in L.texts
        _ptext!(ax, t.x, t.y, t.text; align=(t.halign, :center), fontsize=getfield(_PVENNFONT, t.role) * st.scale,
                color=st.text)
    end
    return L
end

function _venn(::Val{:pbp}, x; sets=nothing, facts=nothing, samples=nothing, scale::Real=260,
               figure=NamedTuple(), axis=NamedTuple(), kwargs...)
    L = _pvennlayout(x, sets, facts, samples)
    fig, ax = _pfigure(L.bbox, scale, figure, axis; m=0.06)
    vennplot!(ax, L; kwargs...)
    return fig
end

# (plain, latex) name of a circle: its condition for sample events (`σ > 0.2`), else the event
function _psetlabel(a, cond, mode::Symbol)
    cond === nothing && return _plabel(a, mode === :none ? :latex : mode)
    plain = string(cond)
    mode === :text && return (plain, "")
    return (plain, something(_conditionlatex(cond), ""))
end

const _HATCH = 0.045      # hatching period, panel units (as in the Makie drawing)

function _pdraw_vennpanel!(ax, L::VennLayout, panel::VennPanel, st)
    n = length(L.sets)
    hw, hh = VENN_BOX
    centres, r, labelpos = _venn_geometry(n)
    X, Y = panel.x, panel.y
    box = _Pt[(X - hw, Y - hh), (X + hw, Y - hh), (X + hw, Y + hh), (X - hw, Y + hh)]
    circles = [_pcircle(X + cx, Y + cy, r; n=120) for (cx, cy) in centres]
    for c in sort!(collect(0:_ncells(n)-1); by=count_ones)
        region = box
        for k in 1:n
            _incircle(c, k) && (region = _intersect_convex(region, circles[k]))
        end
        length(region) < 3 && continue
        bit = UInt32(1) << c
        feasible = L.feasible & bit != 0
        col = !feasible ? st.background :
              panel.target & bit != 0 ? st.highlight :
              panel.given & bit != 0 ? (panel.conditional ? st.given : st.background) : st.excluded
        PBP.poly!(ax, region; color=col)
        feasible && continue
        # impossible cell: hatched, a stripe where (x + y) / period mod 1 < 1/2 (panel coordinates)
        s = [(p[1] - X) + (p[2] - Y) for p in region]
        for k in floor(Int, minimum(s) / _HATCH):ceil(Int, maximum(s) / _HATCH)
            lo, hi = k * _HATCH, (k + 0.5) * _HATCH
            stripe = _keep(_keep(region, p -> (p[1] - X) + (p[2] - Y) - lo), p -> hi - ((p[1] - X) + (p[2] - Y)))
            length(stripe) >= 3 && PBP.poly!(ax, stripe; color=st.impossible)
        end
    end
    for circ in circles
        PBP.lines!(ax, [first.(circ); circ[1][1]], [last.(circ); circ[1][2]]; color=st.stroke, linewidth=st.width)
    end
    PBP.lines!(ax, [X - hw, X + hw, X + hw, X - hw, X - hw], [Y - hh, Y - hh, Y + hh, Y + hh, Y - hh];
               color=st.stroke, linewidth=st.width)
    if st.setlabels
        for (a, cond, (lx, ly)) in zip(L.sets, L.conditions, labelpos)
            _ptext!(ax, X + lx, Y + ly, _psetlabel(a, cond, st.labels); fontsize=_PVENNFONT.set * st.scale,
                    color=st.text)
        end
    end
    if L.cellprobs !== nothing && st.cellvalues !== :none
        for c in 0:_ncells(n)-1
            bit = UInt32(1) << c
            L.feasible & bit == 0 && continue
            intarget = panel.target & bit != 0
            st.cellvalues === :target && !intarget && continue
            ax_, ay_ = _venn_anchor(n, c)
            _ptext!(ax, X + ax_, Y + ay_, _plainnum(L.cellprobs[c+1]); fontsize=_PVENNFONT.cell * st.scale,
                    color=intarget ? st.cellcolor : st.text)
        end
    end
    v = venn_value(L, panel)
    lab = _plabel(panel.prob, st.labels; suffix=v === nothing ? "" : " = " * _plainnum(v))
    _ptext!(ax, X, Y - hh - 0.09, lab; fontsize=_PVENNFONT.label * st.scale, color=st.text)
    return ax
end

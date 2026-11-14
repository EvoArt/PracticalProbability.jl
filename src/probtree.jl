# Probability trees and mosaic plots: the layouts behind `probtree` and
# `mosaic` (drawn by the Makie extension). Both enumerate the same outcomes:
# one level per variable or event, each branch carrying its conditional
# probability P(branch | path so far).

# ---------------------------------------------------------------------------
# Numbers: from a BayesNet, from facts, or from samples
# ---------------------------------------------------------------------------

# A function from a probability expression to its value, or `nothing` when the
# source does not determine it (or there is no source).
function _numbers(; bn=nothing, facts=nothing, samples=nothing)
    count(!isnothing, (bn, facts, samples)) <= 1 ||
        throw(ArgumentError("give numbers from one of bn, facts or samples, not several"))
    fs = facts === nothing ? nothing : facts isa Union{Tuple,AbstractVector} ? facts : (facts,)
    return function (e::PExpr)
        e isa Literal && return Float64(e.value)
        if bn !== nothing
            x = compute(e, bn)
            return x isa Factor ? (isempty(x.vars) ? Float64(x.values[]) : nothing) : Float64(x)
        elseif fs !== nothing
            s = deduce(e, fs...)
            return s.determined ? Float64(s.value) : nothing
        elseif samples !== nothing
            return Float64(compute(e, samples))
        end
        return nothing
    end
end

# ---------------------------------------------------------------------------
# Levels and outcomes
# ---------------------------------------------------------------------------

# The outcomes a level branches into.
function _branches(level::Atom, bn)
    if level isa RV && !isevent(level)
        bn === nothing && throw(ArgumentError("$(level.name) is a variable: pass a BayesNet `bn` for its values, " *
                                              "or use events such as $(level.name)(1)"))
        return Atom[level(v) for v in domain(bn, level)]
    end
    iseventlike(level) || throw(ArgumentError("$(level) cannot be a level of a tree"))
    return Atom[level, complement(level)]
end

# Levels for a query: its events, as variables where a BayesNet gives the values.
function _query_levels(p::Prob, bn)
    levels = Atom[]
    for a in venn_sets(p)
        l = a isa Event && bn !== nothing ? a.rv : a
        l in levels || push!(levels, l)
    end
    if bn !== nothing
        order = topological_order(bn.dag)
        rank(l) = something(findfirst(==(rv(l)), order), length(order) + 1)
        sort!(levels; by=rank)
    end
    return levels
end

_levels(x::Prob, order, bn) = order === nothing ? _query_levels(x, bn) : _atoms(order)
_levels(x::Union{Tuple,AbstractVector}, order, bn) = order === nothing ? _atoms(x) : _atoms(order)
_levels(x::Atom, order, bn) = order === nothing ? Atom[x] : _atoms(order)
_query(x::Prob) = x
_query(x) = nothing

struct _TNode
    atom::Union{Nothing,Atom}        # the outcome on the branch into this node
    prob::Union{Nothing,PExpr}       # P(atom | path so far)
    value::Union{Nothing,Float64}
    path::Vector{Atom}               # outcomes from the root, this one included
    children::Vector{_TNode}
end

function _outcomes(levels::Vector{Atom}, num, bn, g)
    function grow(k, path)
        k > length(levels) && return _TNode[]
        out = _TNode[]
        for b in _branches(levels[k], bn)
            pr = _mkprob(Atom[b], path)
            if g !== nothing && pr isa Prob
                pr = try
                    simplify(pr, g)
                catch
                    pr
                end
            end
            newpath = vcat(path, b)
            push!(out, _TNode(b, pr, num(pr), newpath, grow(k + 1, newpath)))
        end
        return out
    end
    return _TNode(nothing, nothing, 1.0, Atom[], grow(1, Atom[]))
end

_leaves(n::_TNode) = isempty(n.children) ? [n] : reduce(vcat, map(_leaves, n.children))

# Does the event `a` happen on a leaf whose outcomes are `path` (one per level)?
function _happens(a::Atom, levels::Vector{Atom}, path::Vector{Atom})
    a isa SureEvent && return true
    a isa ImpossibleEvent && return false
    k = findfirst(==(a), levels)
    k !== nothing && return path[k] == a
    a isa Complement && return !_happens(a.event, levels, path)
    a isa EventUnion && return any(x -> _happens(x, levels, path), a.parts)
    a isa EventIntersection && return all(x -> _happens(x, levels, path), a.parts)
    if a isa Event
        k = findfirst(==(a.rv), levels)
        k !== nothing && return path[k] == a
    end
    throw(ArgumentError("the tree's levels ($(join(string.(levels), ", "))) do not decide $(a)"))
end

# :target (E ∩ C), :given (C but not E), :excluded (not C), or :none without a query
function _role(q, levels, path)
    q === nothing && return :none
    inC = all(a -> _happens(a, levels, path), q.given)
    inC || return :excluded
    return all(a -> _happens(a, levels, path), q.targets) ? :target : :given
end

_jointvalue(n::_TNode, chain) = begin
    v = 1.0
    for x in chain
        x.value === nothing && return nothing
        v *= x.value
    end
    v
end

# The answer to the query from the leaves, and a caption working it out (plain
# text and LaTeX).
function _answer(q, leaves)
    q === nothing && return (nothing, "", nothing)
    tv = [l.joint for l in leaves if l.role === :target]
    gv = [l.joint for l in leaves if l.role === :given]
    E = _mkprob(vcat(q.targets, q.given), Atom[]; name=q.name)
    C = isempty(q.given) ? nothing : _mkprob(q.given, Atom[]; name=q.name)
    s(x) = sprint(show, x)
    lx(x) = _venn_latex(x)
    if any(isnothing, tv) || any(isnothing, gv)
        C === nothing && return (nothing, s(q), lx(q))
        return (nothing, string(s(q), " = ", s(E), " / ", s(C)),
                string(lx(q), " = \\frac{", lx(E), "}{", lx(C), "}"))
    end
    t = sum(tv; init=0.0)
    C === nothing && return (t, string(s(q), " = ", _num3(t)), string(lx(q), " = ", _num3(t)))
    c = t + sum(gv; init=0.0)
    r = c == 0 ? NaN : t / c
    return (r, string(s(q), " = ", s(E), " / ", s(C), " = ", _num3(t), " / ", _num3(c), " = ", _num3(r)),
            string(lx(q), " = \\frac{", lx(E), "}{", lx(C), "} = \\frac{", _num3(t), "}{", _num3(c), "} = ", _num3(r)))
end

_num3(x) = string(round(x; sigdigits=3))

# "A: σ > 0.2, B: μ > 0" for the sample events among the levels
function _withlegend(caption, samples, levels)
    samples === nothing && return caption
    parts = String[]
    for (e, c) in zip(samples.events, samples.conditions)
        c === nothing || !(e in levels) || push!(parts, string(e.name, ": ", c))
    end
    isempty(parts) && return caption
    leg = string(join(parts, ", "), "  (", samples isa SampleEvents ? ndraws(samples) : "?", " draws)")
    return isempty(caption) ? leg : string(caption, "   [", leg, "]")
end

# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------

"""
    TreeBranch

A branch of a [`TreeLayout`](@ref), from `(x0, y0)` to `(x1, y1)`: the outcome
`atom` at its end, its conditional probability `prob` and that probability's
`value` (or `nothing`).
"""
struct TreeBranch
    x0::Float64
    y0::Float64
    x1::Float64
    y1::Float64
    atom::Atom
    prob::PExpr
    value::Union{Nothing,Float64}
end

"""
    TreeLeaf

A leaf of a [`TreeLayout`](@ref) or a tile of a [`MosaicLayout`](@ref): its
outcomes from the root (`path`), its joint probability (or `nothing`), and its
`role` for the query: `:target`, `:given`, `:excluded`, or `:none` without one.
"""
struct TreeLeaf
    x::Float64
    y::Float64
    path::Vector{Atom}
    joint::Union{Nothing,Float64}
    role::Symbol
end

"""
    TreeLayout

What [`probtree`](@ref) draws: the `levels`, the [`TreeBranch`](@ref)es and
[`TreeLeaf`](@ref)s positioned left to right, the `query` if any, its `result`
worked out from the leaves (or `nothing`), the `caption`, and the bounding box.
Build one with [`tree_layout`](@ref).
"""
struct TreeLayout
    levels::Vector{Atom}
    branches::Vector{TreeBranch}
    leaves::Vector{TreeLeaf}
    query::Union{Nothing,Prob}
    result::Union{Nothing,Float64}
    caption::String
    captionlatex::Union{Nothing,String}
    legend::String
    bbox::NTuple{4,Float64}
end
Base.show(io::IO, L::TreeLayout) =
    print(io, "TreeLayout(", length(L.levels), " levels, ", length(L.leaves), " leaves)")

const TREE_DX = 2.6      # between levels
const TREE_DY = 0.75     # between leaves

"""
    tree_layout(x; order = nothing, bn = nothing, facts = nothing, samples = nothing, dag = nothing)

Work out the probability tree that [`probtree`](@ref) draws, without drawing
it. `x` is a list of levels (variables or events) or a query `P(E | C)`.
Numbers come from one of `bn` (a [`BayesNet`](@ref)), `facts` (as for
[`deduce`](@ref)) or `samples` ([`SampleEvents`](@ref)). Branch labels are
simplified with `dag` (default: `bn`'s graph).
"""
function tree_layout(x; order=nothing, bn=nothing, facts=nothing, samples=nothing, dag=nothing)
    levels = _levels(x, order, bn)
    isempty(levels) && throw(ArgumentError("a tree needs at least one level"))
    q = _query(x)
    num = _numbers(; bn, facts, samples)
    g = dag !== nothing ? dag : bn !== nothing ? bn.dag : nothing
    root = _outcomes(levels, num, bn, g)
    branches = TreeBranch[]
    leaves = TreeLeaf[]
    row = Ref(0)
    function place(n::_TNode, depth, chain)
        if isempty(n.children)
            y = -row[] * TREE_DY
            row[] += 1
            push!(leaves, TreeLeaf(depth * TREE_DX, y, n.path, _jointvalue(n, chain), _role(q, levels, n.path)))
            return y
        end
        ys = Float64[]
        mine = Int[]
        for c in n.children
            yc = place(c, depth + 1, vcat(chain, c))
            push!(ys, yc)
            push!(branches, TreeBranch(depth * TREE_DX, NaN, (depth + 1) * TREE_DX, yc, c.atom, c.prob, c.value))
            push!(mine, length(branches))
        end
        y = (minimum(ys) + maximum(ys)) / 2
        for k in mine       # now the parent's height is known
            b = branches[k]
            branches[k] = TreeBranch(b.x0, y, b.x1, b.y1, b.atom, b.prob, b.value)
        end
        return y
    end
    place(root, 0, _TNode[])
    result, caption, captionlatex = _answer(q, leaves)
    legend = _withlegend("", samples, levels)
    ymin = -(row[] - 1) * TREE_DY
    xmax = length(levels) * TREE_DX + 0.5 + 0.1 * maximum(length(sprint(show, _mkprob(l.path, Atom[]))) + 8 for l in leaves)
    textw = 0.5 * 0.2 * max(length(caption), length(legend))
    below = (isempty(caption) ? 0.0 : 0.6) + (isempty(legend) ? 0.0 : 0.4)
    return TreeLayout(levels, branches, leaves, q, result, caption, captionlatex, legend,
                      (-0.5, max(xmax, textw), ymin - 0.4 - below, 0.6))
end
tree_layout(L::TreeLayout; kwargs...) = L

# ---------------------------------------------------------------------------
# Mosaics
# ---------------------------------------------------------------------------

"""
    MosaicLayout

What [`mosaic`](@ref) draws: the `levels`, one [`TreeLeaf`](@ref) per tile (its
`x`, `y` being the tile's centre) with the tile's rectangle in `rects`
(`(x0, x1, y0, y1)` within the unit square), labels for the first split along
the top (`headers`: outcome and centre x) and for the second split along the
right edge (`sideheaders`: outcome and centre y, from the rightmost column),
the query, its `result` and the `caption`. Build one with [`mosaic_layout`](@ref).
"""
struct MosaicLayout
    levels::Vector{Atom}
    tiles::Vector{TreeLeaf}
    rects::Vector{NTuple{4,Float64}}
    headers::Vector{Tuple{Atom,Float64}}
    sideheaders::Vector{Tuple{Atom,Float64}}
    query::Union{Nothing,Prob}
    result::Union{Nothing,Float64}
    caption::String
    captionlatex::Union{Nothing,String}
    legend::String
end
Base.show(io::IO, L::MosaicLayout) =
    print(io, "MosaicLayout(", length(L.levels), " levels, ", length(L.tiles), " tiles)")

const MOSAIC_GAP = 0.006

"""
    mosaic_layout(x; order = nothing, bn = nothing, facts = nothing, samples = nothing)

Work out the mosaic that [`mosaic`](@ref) draws, without drawing it. Arguments
as for [`tree_layout`](@ref); numbers are required. The unit square is split
left to right by the first level, each strip bottom to top by the second given
the first, then left to right again, so every tile's area is its joint
probability.
"""
function mosaic_layout(x; order=nothing, bn=nothing, facts=nothing, samples=nothing)
    levels = _levels(x, order, bn)
    isempty(levels) && throw(ArgumentError("a mosaic needs at least one level"))
    q = _query(x)
    num = _numbers(; bn, facts, samples)
    root = _outcomes(levels, num, bn, nothing)
    tiles = TreeLeaf[]
    rects = NTuple{4,Float64}[]
    headers = Tuple{Atom,Float64}[]
    side = Tuple{Atom,Float64}[]
    function split(n::_TNode, depth, rect, joint)
        if isempty(n.children)
            x0, x1, y0, y1 = rect
            push!(rects, (x0 + MOSAIC_GAP, x1 - MOSAIC_GAP, y0 + MOSAIC_GAP, y1 - MOSAIC_GAP))
            push!(tiles, TreeLeaf((x0 + x1) / 2, (y0 + y1) / 2, n.path, joint, _role(q, levels, n.path)))
            return
        end
        any(c -> c.value === nothing, n.children) &&
            throw(ArgumentError("a mosaic needs every probability: $(join((sprint(show, c.prob) for c in n.children if c.value === nothing), ", ")) " *
                                "is not determined (pass bn, facts or samples that determine it)"))
        x0, x1, y0, y1 = rect
        lo = 0.0
        depth == 2 && empty!(side)        # keep the rightmost column's (the last one split)
        for c in n.children
            hi = lo + c.value
            # left to right across, top to bottom down (the tree's reading order)
            sub = isodd(depth) ? (x0 + lo * (x1 - x0), x0 + hi * (x1 - x0), y0, y1) :
                  (x0, x1, y1 - hi * (y1 - y0), y1 - lo * (y1 - y0))
            depth == 1 && push!(headers, (c.atom, (sub[1] + sub[2]) / 2))
            depth == 2 && push!(side, (c.atom, (sub[3] + sub[4]) / 2))
            split(c, depth + 1, sub, joint * c.value)
            lo = hi
        end
    end
    split(root, 1, (0.0, 1.0, 0.0, 1.0), 1.0)
    result, caption, captionlatex = _answer(q, tiles)
    return MosaicLayout(levels, tiles, rects, headers, copy(side), q, result, caption, captionlatex,
                        _withlegend("", samples, levels))
end
mosaic_layout(L::MosaicLayout; kwargs...) = L

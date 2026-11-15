# DAG views: node positions, d-separation paths, back-door paths, graph
# surgery and plates. Everything here is plain Julia; `dagplot` in the Makie
# extension draws a `DAGView`.

# ---------------------------------------------------------------------------
# Paths and d-separation
# ---------------------------------------------------------------------------

"""
    DAGPath

A path between two variables in a DAG, ignoring edge directions: its `nodes`
from one end to the other, whether it is `open` given the conditioning set,
the first node that `blocker`s it (or `nothing`), and the `reason` in words.
"""
struct DAGPath
    nodes::Vector{RV}
    open::Bool
    blocker::Union{Nothing,RV}
    reason::String
end

_neighbours(g::DAG, v::RV) = unique(vcat(g.parents[v], children(g, v)))
_iscollider(g::DAG, a::RV, m::RV, b::RV) = a in g.parents[m] && b in g.parents[m]

function _classify(g::DAG, path::Vector{RV}, Z)
    opened = RV[]
    for k in 2:length(path)-1
        a, m, b = path[k-1], path[k], path[k+1]
        if _iscollider(g, a, m, b)
            if m in Z || any(in(Z), descendants(g, m))
                push!(opened, m)
            else
                return DAGPath(path, false, m, "blocked at $(m.name): a collider, and neither it nor " *
                                               "a descendant is conditioned on")
            end
        elseif m in Z
            kind = m in g.parents[a] && m in g.parents[b] ? "a fork" : "a chain"
            return DAGPath(path, false, m, "blocked at $(m.name): $kind, conditioned on")
        end
    end
    reason = isempty(opened) ? "open" :
             "open: the collider $(join((v.name for v in opened), ", ")) is conditioned on " *
             "(itself or a descendant)"
    return DAGPath(path, true, nothing, reason)
end

"""
    dag_paths(g, X, Y, Z = RV[]; backdoor = false, maxpaths = 200) -> Vector{DAGPath}

Every path between the variables `X` and `Y` in `g` (directions ignored, no
variable visited twice, none passing through another variable of `X` or `Y`),
each classified as open or blocked given `Z` by the d-separation rules. A
non-collider on the path blocks it when it is in `Z`. A collider blocks it
unless it or one of its descendants is in `Z`. `X` and `Y` are d-separated
given `Z` exactly when every path is blocked.

With `backdoor = true`, only back-door paths: those that leave `X` along an
edge into `X`. Enumeration stops after `maxpaths` paths.
"""
function dag_paths(g::DAG, X, Y, Z=RV[]; backdoor::Bool=false, maxpaths::Integer=200)
    Xs, Ys, Zs = _rvlist(X), _rvlist(Y), Set(_rvlist(Z))
    out = DAGPath[]
    path = RV[]
    function walk(v)
        length(out) >= maxpaths && return
        if v in Ys
            push!(out, _classify(g, copy(path), Zs))
            return
        end
        for w in _neighbours(g, v)
            (w in path || w in Xs) && continue
            push!(path, w)
            walk(w)
            pop!(path)
        end
    end
    for x in Xs
        empty!(path)
        push!(path, x)
        for w in _neighbours(g, x)
            w in Xs && continue
            backdoor && !(w in g.parents[x]) && continue
            push!(path, w)
            walk(w)
            pop!(path)
        end
    end
    return out
end
dag_paths(g::DAG, s::Independence; kwargs...) = dag_paths(g, s.X, s.Y, s.Z; kwargs...)

"""
    pathstring(g, path) -> String

A path written with its arrows, e.g. `"A → B ← C"`.
"""
function pathstring(g::DAG, path::Vector{RV})
    s = string(path[1].name)
    for k in 2:length(path)
        s *= (path[k-1] in g.parents[path[k]] ? " → " : " ← ") * string(path[k].name)
    end
    return s
end
pathstring(g::DAG, p::DAGPath) = pathstring(g, p.nodes)

# ---------------------------------------------------------------------------
# The view: what to draw, positioned
# ---------------------------------------------------------------------------

"""
    DAGNode

A node of a [`DAGPanel`](@ref): the variable, its position, `shape` (`:circle`,
`:square` for an intervened variable, `:dot` for a model input), `fill`
(`:none`, `:observed` for conditioned-on or observed, `:highlight` for an
adjustment set), whether it is an `endpoint` of the query, and whether it
`blocks` a path.
"""
struct DAGNode
    v::RV
    x::Float64
    y::Float64
    shape::Symbol
    fill::Symbol
    endpoint::Bool
    blocks::Bool
end

"""
    DAGEdge

An edge `from → to` of a [`DAGPanel`](@ref), with a `style`: `:plain`,
`:open` (on an open path), `:blocked` (only on blocked paths) or `:removed`
(cut by an intervention). `bend` is `nothing` for a straight edge, or the
control point of a quadratic curve for an edge that would otherwise pass
through another node.
"""
struct DAGEdge
    from::RV
    to::RV
    style::Symbol
    bend::Union{Nothing,NTuple{2,Float64}}
end
DAGEdge(from, to, style) = DAGEdge(from, to, style, nothing)

# Distance from point p to the segment a–b.
function _segdist(p, a, b)
    dx, dy = b[1] - a[1], b[2] - a[2]
    t = clamp(((p[1] - a[1]) * dx + (p[2] - a[2]) * dy) / (dx^2 + dy^2), 0, 1)
    return hypot(p[1] - a[1] - t * dx, p[2] - a[2] - t * dy)
end

# Where an edge a → b should bend: `nothing` if the straight line is clear,
# else the control point of a quadratic curve bowing away from the nodes in its
# way (to the right when they sit dead centre).
function _bend(a, b, others)
    blockers = [o for o in others if _segdist(o, a, b) < 1.15 * DAG_RADIUS]
    isempty(blockers) && return nothing
    dx, dy = b[1] - a[1], b[2] - a[2]
    len = hypot(dx, dy)
    nx, ny = -dy / len, dx / len
    side = sum((o[1] - a[1]) * nx + (o[2] - a[2]) * ny for o in blockers)
    s = side > 1e-9 ? -1.0 : 1.0
    h = 0.75 + 0.12 * len
    return ((a[1] + b[1]) / 2 + s * h * nx, (a[2] + b[2]) / 2 + s * h * ny)
end

"""
    DAGPlate

A plate: a box around the variables repeated over an index, with its `label`.
"""
struct DAGPlate
    label::String
    x0::Float64
    x1::Float64
    y0::Float64
    y1::Float64
end

"""
    DAGText

Text in a [`DAGView`](@ref), with a `role` (`:title`, `:caption`, `:open`,
`:blocked`, `:note`), horizontal alignment, and a LaTeX form (or `nothing`)
used for drawing when it can be rendered.
"""
struct DAGText
    text::String
    role::Symbol
    x::Float64
    y::Float64
    halign::Symbol
    latex::Union{Nothing,String}
end
DAGText(text, role, x, y, halign) = DAGText(text, role, x, y, halign, nothing)

"""
    DAGPanel

One drawing of a graph: [`DAGNode`](@ref)s, [`DAGEdge`](@ref)s and
[`DAGPlate`](@ref)s, already positioned.
"""
struct DAGPanel
    nodes::Vector{DAGNode}
    edges::Vector{DAGEdge}
    plates::Vector{DAGPlate}
end

"""
    DAGView

Everything [`dagplot`](@ref) draws: one or two [`DAGPanel`](@ref)s, text (titles,
the caption, one line per path), the classified `paths`, and the bounding box
`(xmin, xmax, ymin, ymax)`. Build one with [`dag_view`](@ref).
"""
struct DAGView
    panels::Vector{DAGPanel}
    texts::Vector{DAGText}
    paths::Vector{DAGPath}
    bbox::NTuple{4,Float64}
end

Base.show(io::IO, V::DAGView) =
    print(io, "DAGView(", length(V.panels), length(V.panels) == 1 ? " panel, " : " panels, ",
          length(V.paths), length(V.paths) == 1 ? " path)" : " paths)")

const DAG_RADIUS = 0.3
const _DAG_FONT = (title=0.2, caption=0.2, open=0.15, blocked=0.15, note=0.15, plate=0.15)

# Layered layout. Layers are longest-path depths (roots pulled down next to
# their first child). Variables in the same plates form a band of columns of
# their own, so a plate's box never contains a variable outside the plate.
# Inside each band, a few barycentre sweeps reduce crossings.
function _dag_positions(g::DAG, sig::AbstractDict)
    order = topological_order(g)
    layer = Dict{RV,Int}()
    for v in order
        ps = g.parents[v]
        layer[v] = isempty(ps) ? 1 : 1 + maximum(layer[p] for p in ps)
    end
    for v in order
        isempty(g.parents[v]) || continue
        cs = children(g, v)
        isempty(cs) || (layer[v] = max(1, minimum(layer[c] for c in cs) - 1))
    end
    key(v) = join(string.(get(sig, v, Symbol[])), ",")
    groups = sort(unique(key.(order)))
    nl = maximum(values(layer); init=1)
    cells = Dict{Tuple{Int,String},Vector{RV}}()
    for v in order
        push!(get!(cells, (layer[v], key(v)), RV[]), v)
    end
    width = Dict(k => maximum((length(get(cells, (l, k), RV[])) for l in 1:nl); init=0) for k in groups)
    start = Dict{String,Float64}()
    acc = 0.0
    for k in groups
        start[k] = acc
        acc += width[k]
    end
    xs = Dict{RV,Float64}()
    function place!()
        for ((l, k), vs) in cells
            c = start[k] + (width[k] - 1) / 2
            for (j, v) in enumerate(vs)
                xs[v] = c + j - (length(vs) + 1) / 2
            end
        end
    end
    place!()
    for sweep in 1:8
        for l in (isodd(sweep) ? (1:nl) : (nl:-1:1)), k in groups
            vs = get(cells, (l, k), nothing)
            vs === nothing && continue
            bary(v) = (nb = _neighbours(g, v); isempty(nb) ? xs[v] : sum(xs[w] for w in nb) / length(nb))
            sort!(vs; by=bary)
            c = start[k] + (width[k] - 1) / 2
            for (j, v) in enumerate(vs)
                xs[v] = c + j - (length(vs) + 1) / 2
            end
        end
    end
    centre = (acc - 1) / 2
    return Dict(v => (xs[v] - centre, -(layer[v] - 1.0)) for v in order)
end

function _dag_plates(pos, sig::AbstractDict, labels::AbstractDict)
    out = DAGPlate[]
    names = unique(n for s in values(sig) for n in s)
    for name in names
        members = [v for (v, s) in sig if name in s && haskey(pos, v)]
        isempty(members) && continue
        depth = minimum(something(findfirst(==(name), sig[v])) for v in members)
        inner = maximum(length(sig[v]) for v in members)
        pad = DAG_RADIUS + 0.12 + 0.14 * (inner - depth)
        xs = [pos[v][1] for v in members]
        ys = [pos[v][2] for v in members]
        push!(out, DAGPlate(get(labels, name, string(name)), minimum(xs) - pad, maximum(xs) + pad,
                            minimum(ys) - pad - 0.18, maximum(ys) + pad))
    end
    return out
end

_shiftpanel(p::DAGPanel, dx) =
    DAGPanel([DAGNode(n.v, n.x + dx, n.y, n.shape, n.fill, n.endpoint, n.blocks) for n in p.nodes],
             [DAGEdge(e.from, e.to, e.style, e.bend === nothing ? nothing : (e.bend[1] + dx, e.bend[2])) for e in p.edges],
             [DAGPlate(q.label, q.x0 + dx, q.x1 + dx, q.y0, q.y1) for q in p.plates])

function _panelbox(p::DAGPanel)
    pos = Dict(n.v => (n.x, n.y) for n in p.nodes)
    # a curve's furthest point is halfway between its chord's midpoint and its control point
    peaks = NTuple{2,Float64}[]
    for e in p.edges
        b = e.bend
        b === nothing && continue
        push!(peaks, ((pos[e.from][1] + pos[e.to][1]) / 4 + b[1] / 2, (pos[e.from][2] + pos[e.to][2]) / 4 + b[2] / 2))
    end
    xs = vcat([n.x for n in p.nodes], [q.x0 for q in p.plates], [q.x1 for q in p.plates], first.(peaks))
    ys = vcat([n.y for n in p.nodes], [q.y0 for q in p.plates], [q.y1 for q in p.plates], last.(peaks))
    isempty(xs) && return (-DAG_RADIUS, DAG_RADIUS, -DAG_RADIUS, DAG_RADIUS)
    return (minimum(xs) - DAG_RADIUS - 0.1, maximum(xs) + DAG_RADIUS + 0.1,
            minimum(ys) - DAG_RADIUS - 0.1, maximum(ys) + DAG_RADIUS + 0.1)
end

_textwidth(t::DAGText) = 0.47 * getfield(_DAG_FONT, t.role) * maximum(length, split(t.text, '\n'))

# Put panels side by side with titles, then the caption (plain text and LaTeX)
# and notes underneath.
function _assemble(panels::Vector{DAGPanel}, titles, caption, notes::Vector{Tuple{String,Symbol}}, paths;
                   captionlatex=nothing)
    placed = DAGPanel[]
    texts = DAGText[]
    x = 0.0
    top = -Inf
    bottom = Inf
    for (p, t) in zip(panels, titles)
        b = _panelbox(p)
        lines = isempty(t) ? String[] : split(t, '\n')
        tw = 0.5 * _DAG_FONT.title * maximum(length, lines; init=0)
        slot = max(b[2] - b[1], tw)                      # the panel's width, or its title's
        q = _shiftpanel(p, x + (slot - (b[2] - b[1])) / 2 - b[1])
        b = _panelbox(q)
        push!(placed, q)
        th = 0.27 * length(lines)
        isempty(t) || push!(texts, DAGText(t, :title, (b[1] + b[2]) / 2, b[4] + 0.15 + th / 2, :center))
        x += slot + 0.8
        top = max(top, b[4] + (isempty(t) ? 0.0 : th + 0.2))
        bottom = min(bottom, b[3])
    end
    right = x - 0.8
    y = bottom - 0.35
    if !isempty(caption)
        push!(texts, DAGText(caption, :caption, right / 2, y, :center, captionlatex))
        y -= 0.4
    end
    for (s, role) in notes
        push!(texts, DAGText(s, role, 0.0, y, :left))
        y -= 0.27
    end
    xmin, xmax = 0.0, right
    for t in texts
        w = _textwidth(t)
        lo = t.halign === :left ? t.x : t.halign === :right ? t.x - w : t.x - w / 2
        xmin = min(xmin, lo)
        xmax = max(xmax, lo + w)
    end
    return DAGView(placed, texts, paths, (xmin - 0.1, xmax + 0.1, y + 0.1, top + 0.1))
end

function _dag_panel(g::DAG, pos; sig=Dict{RV,Vector{Symbol}}(), labels=Dict{Symbol,String}(),
                    observed=RV[], highlight=RV[], intervened=RV[], endpoints=RV[], inputs=RV[],
                    blockers=RV[], edgestyle=Dict{Pair{RV,RV},Symbol}())
    nodes = DAGNode[]
    for v in g.nodes
        x, y = pos[v]
        shape = v in intervened ? :square : v in inputs ? :dot : :circle
        fill = v in highlight ? :highlight : v in observed ? :observed : :none
        push!(nodes, DAGNode(v, x, y, shape, fill, v in endpoints, v in blockers))
    end
    edges = [DAGEdge(p, c, get(edgestyle, p => c, :plain),
                     _bend(pos[p], pos[c], [pos[v] for v in g.nodes if v != p && v != c]))
             for (p, c) in PracticalProbability.edges(g)]
    return DAGPanel(nodes, edges, _dag_plates(pos, sig, labels))
end

_plate_sig(::Nothing) = Dict{RV,Vector{Symbol}}()
function _plate_sig(plates::AbstractDict)
    sig = Dict{RV,Vector{Symbol}}()
    for (name, vs) in plates
        for v in _rvlist(vs)
            push!(get!(sig, v, Symbol[]), Symbol(name))
        end
    end
    return sig
end

const _DAG_MAXNOTES = 12

function _pathnotes(g, paths)
    notes = Tuple{String,Symbol}[]
    for p in first(paths, _DAG_MAXNOTES)
        push!(notes, (string(pathstring(g, p), "   (", p.reason, ")"), p.open ? :open : :blocked))
    end
    length(paths) > _DAG_MAXNOTES && push!(notes, ("… and $(length(paths) - _DAG_MAXNOTES) more paths", :note))
    return notes
end

function _pathstyles(paths)
    style = Dict{Pair{RV,RV},Symbol}()
    for p in paths, k in 2:length(p.nodes)
        a, b = p.nodes[k-1], p.nodes[k]
        for e in (a => b, b => a)
            p.open ? (style[e] = :open) : get!(style, e, :blocked)
        end
    end
    return style
end

"""
    dag_view(g::DAG; plates = nothing)
    dag_view(g::DAG, A ⟂ C | B)
    dag_view(g::DAG, P(Y | Do(X)); adjust = nothing)
    dag_view(m::GenerativeModel)

Work out what [`dagplot`](@ref) draws, without drawing it: positions, node and
edge styles, plates, titles, the caption and the classified paths. See
[`dagplot`](@ref) for what each form shows.
"""
function dag_view(g::DAG; plates=nothing, adjust=nothing)
    sig = _plate_sig(plates)
    pos = _dag_positions(g, sig)
    return _assemble([_dag_panel(g, pos; sig)], [""], "", Tuple{String,Symbol}[], DAGPath[])
end

function dag_view(g::DAG, s::Independence; plates=nothing, adjust=nothing)
    sig = _plate_sig(plates)
    pos = _dag_positions(g, sig)
    paths = dag_paths(g, s)
    sep = d_separated(g, s.X, s.Y, s.Z)
    blockers = RV[p.blocker for p in paths if !p.open]
    panel = _dag_panel(g, pos; sig, observed=s.Z, endpoints=vcat(s.X, s.Y), blockers,
                       edgestyle=_pathstyles(paths))
    nopen = count(p -> p.open, paths)
    verdict = sep ? "holds: every path is blocked" :
              "does not hold: $(nopen) open path$(nopen == 1 ? "" : "s")"
    notes = Tuple{String,Symbol}[(verdict, sep ? :blocked : :open)]
    append!(notes, isempty(paths) ? [("no paths connect them", :note)] : _pathnotes(g, paths))
    return _assemble([panel], [""], sprint(show, s), notes, paths; captionlatex=_indep_latex(s))
end

_names_latex(vs) = join((_lx(v.name) for v in vs), ", ")
_indep_latex(s::Independence) =
    string(_names_latex(s.X), " \\perp ", _names_latex(s.Y), isempty(s.Z) ? "" : " \\mid " * _names_latex(s.Z))

# The adjustment set to show: the variables identify's answer sums over that are
# not descendants of X, if they close every back-door path; otherwise pa(X).
function _adjustment(g::DAG, X, Y, W, result)
    desc = descendants(g, X)
    closes(Z) = all(p -> !p.open, dag_paths(g, X, Y, vcat(Z, W); backdoor=true))
    cand = result isa Marginal ? RV[v for v in result.over if !(v in desc) && !(v in X) && v in g.nodes] : RV[]
    closes(cand) && return cand
    pa = unique(RV[p for x in X for p in g.parents[x] if !(p in X)])
    return closes(pa) ? pa : cand
end

function dag_view(g::DAG, p::Prob; plates=nothing, adjust=nothing)
    any(a -> a isa Intervention, p.given) ||
        throw(ArgumentError("dagplot(g, P(…)) draws an interventional query P(Y | Do(X)); " *
                            "for independence, pass a statement such as A ⟂ B | C"))
    X = _rvsof(a for a in p.given if a isa Intervention)
    Y = _rvsof(p.targets)
    W = _rvsof(a for a in p.given if !(a isa Intervention))
    result = identify(p, g)
    Z = adjust === nothing ? _adjustment(g, X, Y, W, result) : _rvlist(adjust)
    sig = _plate_sig(plates)
    pos = _dag_positions(g, sig)
    paths = dag_paths(g, X, Y, vcat(Z, W); backdoor=true)
    left = _dag_panel(g, pos; sig, observed=W, highlight=Z, endpoints=vcat(X, Y),
                      blockers=RV[q.blocker for q in paths if !q.open], edgestyle=_pathstyles(paths))
    cut = Dict{Pair{RV,RV},Symbol}((q => x) => :removed for x in X for q in g.parents[x])
    right = _dag_panel(g, pos; sig, observed=W, intervened=X, endpoints=Y, edgestyle=cut)
    zs = isempty(Z) ? "nothing" : join((v.name for v in Z), ", ")
    xs = join((v.name for v in X), ", ")
    titles = ["back-door paths\nadjusting for $(zs)", "after do($(xs)):\nedges into $(xs) cut"]
    caption = string(sprint(show, p), " = ", sprint(show, result))
    notes = isempty(paths) ? [("no back-door paths from $(xs)", :note)] : _pathnotes(g, paths)
    return _assemble([left, right], titles, caption, notes, paths;
                     captionlatex=_venn_latex(Equation(p, result)))
end

_indexlabel(i) = string(i.name, " = ", _istr(i.lo), ":", _istr(i.hi))

function dag_view(gm::GenerativeModel; plates=nothing, adjust=nothing)
    g = gm.dag
    if plates === nothing
        pl = isdefined(@__MODULE__, :plates) ? getfield(@__MODULE__, :plates)(gm) : Dict()
        sig = Dict{RV,Vector{Symbol}}(v => Symbol[i.name for i in is] for (v, is) in pl if !isempty(is))
        labels = Dict{Symbol,String}(i.name => _indexlabel(i) for is in values(pl) for i in is)
    else
        sig = _plate_sig(plates)
        labels = Dict{Symbol,String}()
    end
    pos = _dag_positions(g, sig)
    panel = _dag_panel(g, pos; sig, labels, observed=gm.observed, inputs=gm.inputs)
    return _assemble([panel], [""], "", Tuple{String,Symbol}[], DAGPath[])
end

dag_view(V::DAGView; kwargs...) = V

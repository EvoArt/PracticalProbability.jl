# Directed acyclic graphs over random variables: d-separation, graph surgery
# for the do-calculus, and the `@dag` macro.

"""
    DAG(edges::Pair...; nodes = RV[])
    DAG(nodes::AbstractVector{RV}, parents::AbstractDict)

A directed acyclic graph over random variables. Edges are `parent => child`;
either side may be a tuple or vector of variables (`(A, B) => C`). `nodes`
adds isolated variables. Usually built with [`@dag`](@ref).
"""
struct DAG
    nodes::Vector{RV}
    parents::Dict{RV,Vector{RV}}
    function DAG(nodes::AbstractVector{RV}, parents::AbstractDict)
        ns = unique(RV[n for n in nodes])
        pa = Dict{RV,Vector{RV}}()
        for n in ns
            pa[n] = RV[]
        end
        for (c, ps) in parents
            c in ns || (push!(ns, c); pa[c] = RV[])
            for p in ps
                p in ns || (push!(ns, p); pa[p] = RV[])
                p in pa[c] || push!(pa[c], p)
            end
        end
        g = new(ns, pa)
        _toposort(g)   # throws on a cycle
        return g
    end
end

_rvlist(x::RV) = RV[x]
_rvlist(x::Union{Tuple,AbstractVector}) = RV[v for v in x]

function DAG(edges::Pair...; nodes=RV[])
    pa = Dict{RV,Vector{RV}}()
    ns = RV[nodes...]
    for (a, b) in edges
        for p in _rvlist(a), c in _rvlist(b)
            push!(get!(pa, c, RV[]), p)
            p in ns || push!(ns, p)
            c in ns || push!(ns, c)
        end
    end
    return DAG(ns, pa)
end

"""
    @dag A → B → C
    @dag begin
        Z → (X, Y)
        X → Y
    end

Build a [`DAG`](@ref). Arrows may be written `→`, `-->` or `=>`, chained, and a
tuple on either side means every node in it. A bare name adds an isolated node.
Names already bound to an [`RV`](@ref) use that variable; other names become
discrete variables.
"""
macro dag(args...)
    stmts = Any[]
    for a in args
        if a isa Expr && a.head === :block
            append!(stmts, filter(x -> !(x isa LineNumberNode), a.args))
        else
            push!(stmts, a)
        end
    end
    edges = Expr[]
    isolated = Any[]
    for s in stmts
        chain = _dag_chain(s)
        if length(chain) == 1
            append!(isolated, map(_rvref, chain[1]))
        end
        for i in 1:length(chain)-1
            push!(edges, :($(Expr(:tuple, map(_rvref, chain[i])...)) =>
                          $(Expr(:tuple, map(_rvref, chain[i+1])...))))
        end
    end
    return :($DAG($(edges...); nodes=$RV[$(isolated...)]))
end

_isarrow(ex) = ex isa Expr && ((ex.head === :call && ex.args[1] in (:→, :(=>))) || ex.head === :-->)
function _dag_chain(ex)
    if _isarrow(ex)
        l, r = ex.head === :--> ? (ex.args[1], ex.args[2]) : (ex.args[2], ex.args[3])
        return vcat(_dag_chain(l), _dag_chain(r))
    elseif ex isa Symbol
        return [Symbol[ex]]
    elseif ex isa Expr && ex.head === :tuple && all(x -> x isa Symbol, ex.args)
        return [Symbol[ex.args...]]
    end
    error("@dag: cannot read `$ex`; write edges like `A → B` or `(A, B) → C`")
end

"""
    nodes(g) -> Vector{RV}
"""
nodes(g::DAG) = copy(g.nodes)

"""
    parents(g, v) -> Vector{RV}
"""
parents(g::DAG, v::RV) = copy(g.parents[v])

"""
    children(g, v) -> Vector{RV}
"""
children(g::DAG, v::RV) = RV[c for c in g.nodes if v in g.parents[c]]

"""
    edges(g) -> Vector{Pair{RV,RV}}
"""
edges(g::DAG) = [p => c for c in g.nodes for p in g.parents[c]]

function _closure(g::DAG, vs, step)
    out = Set{RV}()
    stack = RV[vs...]
    while !isempty(stack)
        v = pop!(stack)
        for w in step(g, v)
            w in out || (push!(out, w); push!(stack, w))
        end
    end
    return out
end

"""
    ancestors(g, vs) -> Set{RV}

Strict ancestors of the variables `vs` (one variable or a collection).
"""
ancestors(g::DAG, vs) = _closure(g, _rvlist(vs), parents)

"""
    descendants(g, vs) -> Set{RV}

Strict descendants of the variables `vs`.
"""
descendants(g::DAG, vs) = _closure(g, _rvlist(vs), children)

"""
    markov_blanket(g, v) -> Set{RV}

Parents, children and the children's other parents of `v`.
"""
function markov_blanket(g::DAG, v::RV)
    mb = Set{RV}(parents(g, v))
    for c in children(g, v)
        push!(mb, c)
        union!(mb, parents(g, c))
    end
    delete!(mb, v)
    return mb
end

function _toposort(g::DAG)
    indeg = Dict(v => length(g.parents[v]) for v in g.nodes)
    ready = sort!(RV[v for v in g.nodes if indeg[v] == 0]; by=v -> findfirst(==(v), g.nodes))
    order = RV[]
    while !isempty(ready)
        v = popfirst!(ready)
        push!(order, v)
        for c in g.nodes
            if v in g.parents[c]
                indeg[c] -= 1
                indeg[c] == 0 && push!(ready, c)
            end
        end
    end
    length(order) == length(g.nodes) || throw(ArgumentError("graph has a directed cycle"))
    return order
end

"""
    topological_order(g) -> Vector{RV}

Every parent comes before its children; ties keep insertion order.
"""
topological_order(g::DAG) = _toposort(g)

"""
    remove_incoming(g, vs) -> DAG

The graph with every edge into `vs` deleted (Pearl's ``G_{\\overline{X}}``):
the graph after intervening on `vs`.
"""
function remove_incoming(g::DAG, vs)
    s = Set(_rvlist(vs))
    return DAG(g.nodes, Dict(c => (c in s ? RV[] : copy(ps)) for (c, ps) in g.parents))
end

"""
    remove_outgoing(g, vs) -> DAG

The graph with every edge out of `vs` deleted (Pearl's ``G_{\\underline{X}}``).
"""
function remove_outgoing(g::DAG, vs)
    s = Set(_rvlist(vs))
    return DAG(g.nodes, Dict(c => RV[p for p in ps if !(p in s)] for (c, ps) in g.parents))
end

"""
    d_separated(g, X, Y, Z = RV[]) -> Bool

Whether `X` and `Y` are d-separated given `Z` in `g` (each a variable or a
collection). Uses the reachability algorithm of Koller & Friedman (Alg. 3.1).
"""
function d_separated(g::DAG, X, Y, Z=RV[])
    Xs, Ys, Zs = Set(_rvlist(X)), Set(_rvlist(Y)), Set(_rvlist(Z))
    isempty(intersect(Xs, Ys)) || return false
    for v in union(Xs, Ys, Zs)
        v in g.nodes || throw(ArgumentError("$(v.name) is not a node of the graph"))
    end
    A = union(ancestors(g, collect(Zs)), Zs)
    tovisit = Tuple{RV,Symbol}[(x, :up) for x in Xs]
    visited = Set{Tuple{RV,Symbol}}()
    while !isempty(tovisit)
        y, d = pop!(tovisit)
        (y, d) in visited && continue
        push!(visited, (y, d))
        if !(y in Zs) && y in Ys
            return false
        end
        if d === :up && !(y in Zs)
            for p in g.parents[y]
                push!(tovisit, (p, :up))
            end
            for c in children(g, y)
                push!(tovisit, (c, :down))
            end
        elseif d === :down
            if !(y in Zs)
                for c in children(g, y)
                    push!(tovisit, (c, :down))
                end
            end
            if y in A
                for p in g.parents[y]
                    push!(tovisit, (p, :up))
                end
            end
        end
    end
    return true
end

# ---------------------------------------------------------------------------
# Independence statements:  A ⟂ B | C
# ---------------------------------------------------------------------------

"""
    Independence(X, Y, Z)
    X ⟂ Y | Z

The statement "`X` is independent of `Y` given `Z`". `⟂` is typed `\\perp<tab>`.
Check it against a graph with [`holds`](@ref).
"""
struct Independence
    X::Vector{RV}
    Y::Vector{RV}
    Z::Vector{RV}
end
_rvs(x::Atom) = rvs(x)
_rvs(xs::Union{Tuple,AbstractVector}) = unique(RV[v for x in xs for v in rvs(x)])

⟂(a, b::Given) = Independence(_rvs(a), _rvs(b.targets), _rvs(b.given))
⟂(a, b::Union{Atom,Tuple}) = Independence(_rvs(a), _rvs(b), RV[])

Base.:(==)(a::Independence, b::Independence) =
    issetequal(a.Z, b.Z) && ((issetequal(a.X, b.X) && issetequal(a.Y, b.Y)) ||
                             (issetequal(a.X, b.Y) && issetequal(a.Y, b.X)))
# consistent with ==: the two sides are unordered
Base.hash(a::Independence, h::UInt) = hash(Set(a.Z), hash(hash(Set(a.X)) + hash(Set(a.Y)), hash(:Independence, h)))


"""
    holds(statement::Independence, g::DAG) -> Bool

Whether the graph implies the independence, by d-separation.
"""
holds(s::Independence, g::DAG) = d_separated(g, s.X, s.Y, s.Z)

function _independencies(g::DAG; max_conditioning::Integer=2)
    out = Independence[]
    ns = g.nodes
    for i in eachindex(ns), j in i+1:length(ns)
        a, b = ns[i], ns[j]
        others = RV[v for v in ns if v != a && v != b]
        found = Vector{Set{RV}}()
        for k in 0:min(max_conditioning, length(others))
            for Z in _combinations(others, k)
                Zs = Set(Z)
                any(f -> issubset(f, Zs), found) && continue
                if d_separated(g, a, b, Z)
                    push!(found, Zs)
                    push!(out, Independence(RV[a], RV[b], Z))
                end
            end
        end
    end
    return out
end

function _combinations(xs::Vector{T}, k::Integer) where {T}
    k == 0 && return [T[]]
    k > length(xs) && return Vector{T}[]
    out = Vector{T}[]
    for i in 1:length(xs)-k+1
        for rest in _combinations(xs[i+1:end], k - 1)
            push!(out, vcat(T[xs[i]], rest))
        end
    end
    return out
end

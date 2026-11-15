# What a graph implies about the observed distribution, and when two graphs
# imply the same: a basis of conditional independences (the ordered local
# Markov property), the functional (Verma) constraints of an ADMG (Tian &
# Pearl 2002), and Markov equivalence of DAGs (Verma & Pearl 1990), with the
# CPDAG (essential graph) by Meek's rules. Symbolic only: no tests on data.

# ---------------------------------------------------------------------------
# A basis of conditional independences
# ---------------------------------------------------------------------------


# The district of v in G[S], and its Markov blanket there.
function _district_in(g::ADMG, S, v::RV)
    sub = subgraph(g, S)
    for d in districts(sub)
        v in d && return (sub, d)
    end
    return (sub, RV[v])
end

"""
    markov_basis(g) -> Independencies

A basis of the conditional independences of a DAG or ADMG: the ordered local
Markov property. In a topological order, each `V` is independent of its other
predecessors given its Markov blanket among them (in a DAG, its parents;
in an ADMG, its district there and the district's parents). Every
independence `g` implies follows from these by the graphoid axioms; an ADMG
also implies the equality constraints of [`verma_constraints`](@ref).
"""
function markov_basis(g::Union{DAG,ADMG})
    G = _asadmg(g)
    order = topological_order(G)
    out = Independence[]
    for (i, v) in enumerate(order)
        S = order[1:i]
        sub, d = _district_in(G, S, v)
        mb = Set{RV}(w for w in d if w != v)
        for w in d, p in parents(sub, w)
            p == v || push!(mb, p)
        end
        rest = RV[w for w in S if w != v && !(w in mb)]
        isempty(rest) && continue
        push!(out, Independence(RV[v], rest, RV[w for w in order if w in mb]))
    end
    return Independencies(out)
end

# ---------------------------------------------------------------------------
# Functional (Verma) constraints
# ---------------------------------------------------------------------------

"""
    FunctionalConstraint

An equality constraint of an ADMG: `expr`, built from the observed
distribution, depends only on the variables `on` and not on `noton`.
`holds(c, bn)` checks it numerically. Returned by
[`verma_constraints`](@ref).
"""
struct FunctionalConstraint
    expr::PExpr
    on::Vector{RV}
    noton::Vector{RV}
end
function Base.show(io::IO, c::FunctionalConstraint)
    print(io, c.expr, " does not depend on ", join((v.name for v in c.noton), ", "))
end
function Base.show(io::IO, ::MIME"text/plain", cs::AbstractVector{FunctionalConstraint})
    print(io, length(cs), " functional constraint", length(cs) == 1 ? "" : "s", ":")
    for c in cs
        print(io, "\n  ", c)
    end
end

"""
    verma_constraints(g::ADMG) -> Vector{FunctionalConstraint}

The equality constraints a hidden-variable model imposes beyond conditional
independence, found by Tian & Pearl's (2002) procedure: in each prefix of a
topological order, the c-factor `Q[S]` of the district `S` of the last
variable is summed over non-ancestral parts, and each resulting `Q[E]` can
depend only on `E` and its parents. For the Verma graph
`V₁ → V₂ → V₃ → V₄, V₂ ↔ V₄` this is `∑_{v₂} P(v₄ | v₁, v₂, v₃) P(v₂ | v₁)`
not depending on `v₁`. A DAG has none. Some constraints listed may also follow
from independences; [`markov_basis`](@ref) lists those.
"""
function verma_constraints(g::Union{DAG,ADMG})
    G = _asadmg(g)
    order = topological_order(G)
    out = FunctionalConstraint[]
    seen = Set{Tuple{Vector{RV},Vector{RV}}}()
    for (i, v) in enumerate(order)
        S = order[1:i]
        sub, H = _district_in(G, S, v)
        Q = mul((Prob(Atom[w], Atom[order[1:findfirst(==(w), order)-1]...]) for w in H)...)
        _verma_find!(out, seen, sub, H, Q, v, order)
    end
    return out
end

# Every proper ancestral subset D ∋ v of G[C], its c-factor, and the district E of v in G[D].
function _verma_find!(out, seen, sub::ADMG, C::Vector{RV}, QC::PExpr, v::RV, order)
    GC = subgraph(sub, C)
    rest = RV[w for w in C if w != v]
    for k in 0:length(rest)-1, part in _combinations(rest, k)
        D = RV[w for w in C if w == v || w in part]
        _isancestral(GC.dag, D) || continue
        QD = marg(RV[w for w in C if !(w in D)], QC)
        GD = subgraph(sub, D)
        E = only(d for d in districts(GD) if v in d)
        QE = _cfactor_of(QD, D, E, order)
        allowed = Set{RV}(E)
        for w in E, p in parents(sub, w)
            push!(allowed, p)
        end
        free = freevars(QE)
        noton = sort!(RV[w for w in free if !(w in allowed)])
        if !isempty(noton)
            key = (sort(E), noton)
            if !(key in seen)
                push!(seen, key)
                push!(out, FunctionalConstraint(QE, sort!(RV[w for w in free if w in allowed]), noton))
            end
        end
        length(E) > 1 && _verma_find!(out, seen, sub, E, QE, v, order)
    end
end

_isancestral(g::DAG, D) = all(p -> p in D, (p for w in D for p in g.parents[w]))

# Q[E] for a district E of G[D], from Q[D] (Tian & Pearl 2002, Lemma 4):
# ∏_{Dⱼ ∈ E} Q[D⁽ʲ⁾] / Q[D⁽ʲ⁻¹⁾], Q[D⁽ʲ⁾] = ∑_{D ∖ D⁽ʲ⁾} Q[D].
function _cfactor_of(QD::PExpr, D::Vector{RV}, E::Vector{RV}, order)
    issetequal(D, E) && return QD
    Ds = RV[w for w in order if w in D]
    prefix(j) = marg(Ds[j+1:end], QD)
    return mul((divide(prefix(j), j == 1 ? Literal(1) : prefix(j - 1)) for (j, w) in enumerate(Ds) if w in E)...)
end

"""
    holds(c::FunctionalConstraint, bn; atol = 1e-10) -> Bool

Whether the constraint holds in a network (whose nodes may include hidden
variables): `c.expr`, computed on `bn`, is constant in `c.noton`.
"""
function holds(c::FunctionalConstraint, nets::_Nets; atol::Real=1e-10)
    f = compute(c.expr, nets)
    f isa Factor || return true
    dims = [k for (k, v) in enumerate(f.vars) if v in c.noton]
    isempty(dims) && return true
    A = f.values
    for d in dims

        ref = selectdim(A, d, 1:1)
        all(isapprox.(A, ref; atol) .| (isnan.(A) .& isnan.(ref))) || return false
    end
    return true
end

# ---------------------------------------------------------------------------
# Markov equivalence
# ---------------------------------------------------------------------------

"""
    CPDAG

A completed partially directed acyclic graph (essential graph): the edges
every DAG in a Markov equivalence class orients the same way (`directed`),
and the rest (`undirected`). Two DAGs are Markov equivalent iff their CPDAGs
are equal. Build with [`cpdag`](@ref).
"""
struct CPDAG
    nodes::Vector{RV}
    directed::Set{Pair{RV,RV}}
    undirected::Set{Set{RV}}
end
Base.:(==)(a::CPDAG, b::CPDAG) = issetequal(a.nodes, b.nodes) && a.directed == b.directed && a.undirected == b.undirected
Base.hash(a::CPDAG, h::UInt) = hash(a.undirected, hash(a.directed, hash(Set(a.nodes), h)))
function Base.show(io::IO, c::CPDAG)
    print(io, "CPDAG(", length(c.nodes), " nodes, ", length(c.directed), " directed, ",
          length(c.undirected), " undirected edges)")
end
function Base.show(io::IO, ::MIME"text/plain", c::CPDAG)
    print(io, "CPDAG with ", length(c.nodes), " nodes:")
    for e in sort!(collect(c.directed); by=e -> (e.first.name, e.second.name))
        print(io, "\n  ", e.first.name, " → ", e.second.name)
    end
    for e in sort!([sort!(collect(u)) for u in c.undirected]; by=u -> (u[1].name, u[2].name))
        print(io, "\n  ", e[1].name, " — ", e[2].name)
    end
end
edges(c::CPDAG) = (sort!(collect(c.directed); by=e -> (e.first.name, e.second.name)),
                   sort!([Tuple(sort!(collect(u))) for u in c.undirected]))

_adjacent(g::DAG, a::RV, b::RV) = a in g.parents[b] || b in g.parents[a]

"""
    vstructures(g::DAG) -> Vector{Tuple{RV,RV,RV}}

The v-structures (unshielded colliders) `a → c ← b` with `a`, `b` not
adjacent, as `(a, c, b)` with `a < b`.
"""
function vstructures(g::DAG)
    out = Tuple{RV,RV,RV}[]
    for c in g.nodes
        ps = sort(g.parents[c])
        for i in eachindex(ps), j in i+1:length(ps)
            _adjacent(g, ps[i], ps[j]) || push!(out, (ps[i], c, ps[j]))
        end
    end
    return out
end

"""
    cpdag(g::DAG) -> CPDAG

The essential graph of the Markov equivalence class of `g`: its skeleton,
with the v-structures oriented and Meek's rules 1–3 applied until nothing
more can be oriented.
"""
function cpdag(g::DAG)
    dir = Set{Pair{RV,RV}}()
    und = Set{Set{RV}}(Set([p, c]) for c in g.nodes for p in g.parents[c])
    for (a, c, b) in vstructures(g)
        for x in (a, b)
            delete!(und, Set([x, c]))
            push!(dir, x => c)
        end
    end
    adj(a, b) = Set([a, b]) in und || (a => b) in dir || (b => a) in dir
    isund(a, b) = Set([a, b]) in und
    orient!(a, b) = (delete!(und, Set([a, b])); push!(dir, a => b))
    changed = true
    while changed
        changed = false
        for e in collect(und)
            x, y = collect(e)
            for (b, c) in ((x, y), (y, x))
                isund(b, c) || continue
                # R1: a → b — c, a and c not adjacent  ⇒  b → c
                r1 = any(d -> d.second == b && d.first != c && !adj(d.first, c), dir)
                # R2: b → a → c and b — c  ⇒  b → c
                r2 = any(d -> d.first == b && (d.second => c) in dir, dir)
                # R3: b — a₁ → c, b — a₂ → c, a₁ and a₂ not adjacent, b — c  ⇒  b → c
                as = [d.first for d in dir if d.second == c && isund(b, d.first)]
                r3 = any(adj(as[i], as[j]) == false for i in eachindex(as) for j in i+1:length(as))
                if r1 || r2 || r3
                    orient!(b, c)
                    changed = true
                    break
                end
            end
        end
    end
    return CPDAG(copy(g.nodes), dir, und)
end

"""
    markov_equivalent(g₁::DAG, g₂::DAG) -> Bool

Whether two DAGs imply the same conditional independences: the same
skeleton and the same v-structures (Verma & Pearl 1990).
"""
function markov_equivalent(g1::DAG, g2::DAG)
    issetequal(g1.nodes, g2.nodes) || return false
    skel(g) = Set(Set([p, c]) for c in g.nodes for p in g.parents[c])
    return skel(g1) == skel(g2) && issetequal(vstructures(g1), vstructures(g2))
end

"""
    equivalence_class(g::DAG; maxundirected = 16) -> Vector{DAG}

Every DAG Markov equivalent to `g`: the orientations of the undirected edges
of its [`cpdag`](@ref) that create no cycle and no new v-structure.
"""
function equivalence_class(g::DAG; maxundirected::Integer=16)
    c = cpdag(g)
    und = [Tuple(sort!(collect(u))) for u in c.undirected]
    length(und) > maxundirected && throw(ArgumentError(string(length(und), " undirected edges: too many orientations")))
    out = DAG[]
    vs = Set(vstructures(g))
    for bits in 0:(2^length(und) - 1)
        pa = Dict{RV,Vector{RV}}(v => RV[] for v in c.nodes)
        for d in c.directed
            push!(pa[d.second], d.first)
        end
        for (k, (a, b)) in enumerate(und)
            isodd(bits >> (k - 1)) ? push!(pa[a], b) : push!(pa[b], a)
        end
        h = try
            DAG(c.nodes, pa)
        catch
            continue                     # a cycle
        end
        Set(vstructures(h)) == vs && push!(out, h)
    end
    return out
end

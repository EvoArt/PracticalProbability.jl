# Rewrite rules.
#
# Two kinds of operation:
#
# * Explicit transformations the user asks for: `chainrule`, `bayes`,
#   `marginalize`, `condition_on`, `factorize`, `expand`, `identify`, and the
#   event rules in eventrules.jl.
# * `simplify`, a fixed-point rewrite with a set of local rules. Every local
#   rule returns `(new_subexpression, justification)` or `nothing`, so a run can
#   be recorded as a `Derivation`: each line carries the rule that produced it.

# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------

"""
    Derivation

A chain of equal expressions, each with the rule that produced it from the line
before. Returned by `simplify(…; trace = true)`, [`identify`](@ref) and
friends, and built by hand with [`derivation`](@ref) and [`apply!`](@ref).
Displays as a table in the REPL and as an `aligned` block in LaTeX.
"""
struct Derivation
    exprs::Vector{PExpr}
    rules::Vector{String}
end

"""
    derivation(expr) -> Derivation

Start a derivation from `expr`. Extend it with [`apply!`](@ref).
"""
derivation(e::PExpr) = Derivation(PExpr[e], String["start"])

Base.last(d::Derivation) = d.exprs[end]
Base.first(d::Derivation) = d.exprs[1]
Base.length(d::Derivation) = length(d.exprs)

function _push!(d::Derivation, e::PExpr, rule::AbstractString)
    e == last(d) && return d
    push!(d.exprs, e)
    push!(d.rules, String(rule))
    return d
end

"""
    apply!(d::Derivation, f, label = string(f); kwargs...) -> Derivation

Apply `f` to the last line of `d`. If `f` returns a `Derivation` (as
`simplify(…; trace = true)` does) its steps are appended; otherwise the result
is added as one step labelled `label`.
"""
function apply!(d::Derivation, f, label::AbstractString=string(nameof(f)); kwargs...)
    r = f(last(d); kwargs...)
    if r isa Derivation
        for (e, rule) in zip(r.exprs[2:end], r.rules[2:end])
            _push!(d, e, rule)
        end
    else
        _push!(d, r, label)
    end
    return d
end

Equation(d::Derivation) = Equation(first(d), last(d))

# ---------------------------------------------------------------------------
# Explicit transformations
# ---------------------------------------------------------------------------

_aslist(x::Atom) = Atom[x]
_aslist(x) = Atom[a for a in x]
_rvsof(atoms) = unique(RV[v for a in atoms for v in rvs(a)])

# Sum `body` over the atoms in `bound`: an ordinary ∑/∫ for plain variables,
# and for an event E the two cases E and Eᶜ (law of total probability).
function _sumover(bound, body::PExpr)
    vs = RV[]
    terms = PExpr[body]
    for a in bound
        if iseventlike(a)
            terms = vcat(terms, PExpr[substitute(t, a => complement(a)) for t in terms])
        else
            v = freerv(a)
            v === nothing || push!(vs, v)
        end
    end
    return marg(vs, add(terms...))
end

"""
    chainrule(P(A, B, C | D); order = targets) -> PExpr

The chain rule: `P(A, B, C | D) = P(A | B, C, D) P(B | C, D) P(C | D)`, peeling
targets off in `order`.
"""
function chainrule(p::Prob; order=p.targets)
    order = _aslist(order)
    issetequal(order, p.targets) || throw(ArgumentError("order must list exactly the targets of $(p)"))
    return mul((_like(p, Atom[order[i]], vcat(order[i+1:end], p.given)) for i in eachindex(order))...)
end

"""
    bayes(P(A | B, C); swap = [B, C], expand = false) -> PExpr

Bayes' rule. The conditioning variables in `swap` (default: every
observational one; `do(⋅)` terms never move) change places with the targets,
and the rest stay conditioned on:
`bayes(P(A | B, C); swap = B) = P(B | A, C) P(A | C) / P(B | C)`.

With `expand = true` the denominator is written out by the law of total
probability: `∑_A P(B | A, C) P(A | C)` for a variable `A`, and
`P(B | A, C) P(A | C) + P(B | Aᶜ, C) P(Aᶜ | C)` for an event.
"""
function bayes(p::Prob; swap=nothing, expand::Bool=false)
    isempty(p.given) && throw(ArgumentError("Bayes' rule needs a conditioning variable"))
    B = swap === nothing ? Atom[g for g in p.given if !(g isa Intervention)] : _aslist(swap)
    isempty(B) && throw(ArgumentError("nothing to swap: only do(⋅) terms on the right of the bar"))
    all(b -> b in p.given && !(b isa Intervention), B) ||
        throw(ArgumentError("swap must be observational conditioning variables of $(p)"))
    A = p.targets
    C = Atom[g for g in p.given if !(g in B)]
    num = mul(_like(p, B, vcat(A, C)), _like(p, A, C))
    den = expand ? _sumover(A, num) : _like(p, B, C)
    return divide(num, den)
end

"""
    marginalize(P(A | C), B) -> PExpr

The law of total probability: `P(A | C) = ∑_B P(A, B | C)`, or for an event
`B`, `P(A | C) = P(A ∩ B | C) + P(A ∩ Bᶜ | C)`.
"""
function marginalize(p::Prob, B)
    bs = _aslist(B)
    any(v -> v in _allvars(p), _rvsof(bs)) &&
        throw(ArgumentError("cannot marginalise over a variable already in $(p)"))
    return _sumover(bs, _like(p, vcat(p.targets, bs), p.given))
end

"""
    condition_on(P(A | C), B) -> PExpr
    total_probability(P(A | C), B) -> PExpr

Condition on `B` and average over it: `P(A | C) = ∑_B P(A | B, C) P(B | C)`,
or for an event `B`, `P(A | B, C) P(B | C) + P(A | Bᶜ, C) P(Bᶜ | C)`.
"""
function condition_on(p::Prob, B)
    bs = _aslist(B)
    any(v -> v in _allvars(p), _rvsof(bs)) &&
        throw(ArgumentError("cannot condition on a variable already in $(p)"))
    return _sumover(bs, mul(_like(p, p.targets, vcat(bs, p.given)), _like(p, bs, p.given)))
end
const total_probability = condition_on

"""
    factorize(g::DAG; exclude = RV[], name = :P) -> PExpr

The joint distribution implied by `g`, as the product of each node given its
parents, in topological order. Nodes in `exclude` (fixed inputs) get no factor
but still appear as parents.
"""
function factorize(g::DAG; exclude=RV[], name::Symbol=:P)
    ex = Set(_rvlist(exclude))
    return mul((Prob(Atom[v], Atom[p for p in g.parents[v]], name)
                for v in topological_order(g) if !(v in ex))...)
end

"""
    joint(g::DAG; name = :P) -> Prob

The joint distribution of every node, `P(V₁, …, Vₙ)`.
"""
joint(g::DAG; name::Symbol=:P) = Prob(Atom[v for v in topological_order(g)], Atom[], name)

# ---------------------------------------------------------------------------
# Local simplification rules
# ---------------------------------------------------------------------------

# A graph describes one distribution: its independencies say nothing about
# another distribution (a proposal q(x | y) is an arbitrary kernel), so each
# graph is keyed by the name of the distribution it belongs to.
struct SimplifyContext
    dags::Dict{Symbol,DAG}
    combine::Bool
    facts::Facts
end
_dags(::Nothing) = Dict{Symbol,DAG}()
_dags(g::DAG) = Dict{Symbol,DAG}(:P => g)
_dags(d::AbstractDict) = Dict{Symbol,DAG}(Symbol(k) => _asdag(v) for (k, v) in d)
_asdag(g::DAG) = g     # admg.jl adds ADMGs (as their canonical DAG with latent nodes)
_graph(ctx::SimplifyContext, p::Prob) = get(ctx.dags, p.name, nothing)

_atomset(xs) = Set{Atom}(xs)

# P(A, B | A) = P(B | A);  P(A | A) = 1;  P(Aᶜ | A) = 0;  P(A ∪ B | A) = 1
function rule_trivial(p::Prob, ::SimplifyContext)
    for t in p.targets
        any(g -> _iscomplementpair(t, g), p.given) &&
            return (Literal(0), "an event has probability 0 given its complement")
    end
    # Only an identical atom: P(X | X = 1) is an indicator of X, not 1.
    keep = Atom[t for t in p.targets if !(t in p.given) &&
                                       !(t isa EventUnion && any(x -> x in p.given, t.parts))]
    length(keep) == length(p.targets) && return nothing
    isempty(keep) && return (Literal(1), "given an event, it happens with probability 1")
    return (_like(p, keep, p.given), "drop targets that the conditioning implies")
end
rule_trivial(::PExpr, ::SimplifyContext) = nothing

# Independence: drop conditioning atoms that are d-separated from the
# targets given the rest (with do(⋅) terms: do-calculus rule 1).
function rule_independence(p::Prob, ctx::SimplifyContext)
    g = _graph(ctx, p)
    g === nothing && return nothing
    isempty(p.given) && return nothing
    X = _rvsof(a for a in p.given if a isa Intervention)
    W = Atom[a for a in p.given if !(a isa Intervention)]
    T = _rvsof(p.targets)
    all(v -> v in g.nodes, vcat(T, X, _rvsof(W))) || return nothing
    gx = isempty(X) ? g : remove_incoming(g, X)
    for w in W
        rest = _rvsof(a for a in W if a != w)
        wv = rvs(w)
        (isempty(wv) || any(v -> v in rest || v in T, wv)) && continue
        if d_separated(gx, T, wv, vcat(X, rest))
            newgiven = Atom[a for a in p.given if a != w]
            why = isempty(X) ? "$(_str(T)) ⟂ $(_str(wv))$(_givenstr(rest)) (d-separation)" :
                  "do-calculus rule 1: $(_str(T)) ⟂ $(_str(wv))$(_givenstr(vcat(X, rest))) in G[X̄]"
            return (_like(p, p.targets, newgiven), why)
        end
    end
    return nothing
end
rule_independence(::PExpr, ::SimplifyContext) = nothing

# do-calculus rules 2 and 3.
function rule_docalculus(p::Prob, ctx::SimplifyContext)
    g = _graph(ctx, p)
    g === nothing && return nothing
    ints = Intervention[a for a in p.given if a isa Intervention]
    isempty(ints) && return nothing
    T = _rvsof(p.targets)
    W = _rvsof(a for a in p.given if !(a isa Intervention))
    all(v -> v in g.nodes, vcat(T, W, _rvsof(ints))) || return nothing
    for z in ints
        Z = rv(z)
        X = RV[rv(a) for a in ints if a != z]
        gX = remove_incoming(g, X)
        # Rule 3: delete the action.  Z(W) = Z-nodes that are not ancestors of W in G[X̄]
        anc = ancestors(gX, W)
        g3 = Z in anc ? gX : remove_incoming(gX, Z)
        if d_separated(g3, T, Z, vcat(X, W))
            return (_like(p, p.targets, Atom[a for a in p.given if a != z]),
                    "do-calculus rule 3: $(_str(T)) ⟂ $(Z.name)$(_givenstr(vcat(X, W))) in G[$(_bar(X))$(Z in anc ? "" : "$(Z.name)̄")], delete do($(Z.name))")
        end
        # Rule 2: exchange action for observation.
        g2 = remove_outgoing(gX, Z)
        if d_separated(g2, T, Z, vcat(X, W))
            newgiven = Atom[a == z ? z.target : a for a in p.given]
            return (_like(p, p.targets, newgiven),
                    "do-calculus rule 2: $(_str(T)) ⟂ $(Z.name)$(_givenstr(vcat(X, W))) in G[$(_bar(X))$(Z.name)̲], do($(Z.name)) → $(Z.name)")
        end
    end
    return nothing
end
rule_docalculus(::PExpr, ::SimplifyContext) = nothing

_str(T::Vector{RV}) = join((v.name for v in T), ", ")
_givenstr(Z::Vector{RV}) = isempty(Z) ? "" : " | " * _str(Z)
_bar(X::Vector{RV}) = join(("$(x.name)̄" for x in X), "")

_factors(e::PExpr) = e isa Product ? e.factors : PExpr[e]

# Sums: eliminate summed variables that appear as a target of exactly one
# factor and nowhere else; pull factors that don't involve the summed
# variables out of the sum.
function rule_marginal(m::Marginal, ctx::SimplifyContext)
    fs = _factors(m.body)
    for v in m.over
        holders = findall(f -> mentions(f, v), fs)
        length(holders) == 1 || continue
        f = fs[holders[1]]
        f isa Prob || continue
        ti = findfirst(==(v), f.targets)      # as a free target, not an event
        ti === nothing && continue
        any(a -> v in rvs(a), f.given) && continue
        newtargets = Atom[t for t in f.targets if t != v]
        newf = isempty(newtargets) ? Literal(1) : _like(f, newtargets, f.given)
        newfs = copy(fs)
        newfs[holders[1]] = newf
        over = RV[w for w in m.over if w != v]
        why = isempty(newtargets) ? "probabilities sum to one over $(v.name)" :
              "marginalise $(v.name) out of $(f)"
        return (marg(over, mul(newfs...)), why)
    end
    # Two factors share a summed variable and the chain rule merges them into
    # one with that variable as a target, which the step above then sums out:
    # ∑_A P(B | A) P(A) = ∑_A P(A, B) = P(B). (Variable elimination.)
    for v in m.over
        holders = findall(f -> mentions(f, v), fs)
        length(holders) == 2 || continue
        f1, f2 = fs[holders[1]], fs[holders[2]]
        why = "chain rule: $(f1) $(f2) = "
        new = _chain_merge(f1, f2)
        new === nothing && (new = _chain_merge(f2, f1))
        if new === nothing         # with the distribution's graph, add conditioning first
            out = _graph_merge(f1, f2, ctx)
            out === nothing && (out = _graph_merge(f2, f1, ctx))
            out === nothing || ((new, why) = out)
        end
        (new === nothing || !(v in new.targets)) && continue
        rest = PExpr[fs[k] for k in eachindex(fs) if !(k in holders)]
        return (marg(m.over, mul(new, rest...)), why * sprint(show, new))
    end
    # pull out factors free of the summed variables
    if length(fs) > 1
        out = PExpr[f for f in fs if !any(v -> mentions(f, v), m.over)]
        if !isempty(out) && length(out) < length(fs)
            inn = PExpr[f for f in fs if any(v -> mentions(f, v), m.over)]
            return (mul(mul(out...), marg(m.over, mul(inn...))),
                    "move factors without $(_str(m.over)) outside the sum")
        end
    end
    return nothing
end
rule_marginal(::PExpr, ::SimplifyContext) = nothing

# Chain rule backwards: P(A | B, C) P(B | C) = P(A, B | C), same distribution only.
function _chain_merge(f1, f2)
    (f1 isa Prob && f2 isa Prob && f1.name === f2.name) || return nothing
    B = _atomset(f2.targets)
    G1 = _atomset(f1.given)
    issubset(B, G1) || return nothing
    setdiff(G1, B) == _atomset(f2.given) || return nothing
    return _like(f1, vcat(f1.targets, f2.targets), f2.given)
end

# The chain rule after widening one factor's conditioning set by
# d-separation. For f1 = P(T1 | G1), f2 = P(T2 | G2) with T2 ⊆ G1: if G2 has
# variables E beyond G1∖T2 and T1 ⟂ E | G1, read f1 as P(T1 | G1, E); if
# G1∖T2 has variables E beyond G2 and T2 ⟂ E | G2, read f2 as P(T2 | G2, E).
# E.g. under X1 → X2 → Y2: P(Y2 | X2) P(X2 | X1) = P(Y2, X2 | X1).
function _graph_merge(f1, f2, ctx::SimplifyContext)
    (f1 isa Prob && f2 isa Prob && f1.name === f2.name) || return nothing
    g = _graph(ctx, f1)
    g === nothing && return nothing
    all(v -> v in g.nodes, _rvsof(vcat(f1.targets, f1.given, f2.targets, f2.given))) || return nothing
    any(a -> a isa Intervention, vcat(f1.given, f2.given)) && return nothing
    T2 = _atomset(f2.targets)
    G1 = _atomset(f1.given)
    G2 = _atomset(f2.given)
    T1v = _rvsof(f1.targets)
    T2v = _rvsof(f2.targets)
    if !issubset(T2, G1)
        # f2's targets go beyond f1's conditioning set: widen f1's conditioning
        # to everything f2 is about, P(T1 | G1) = P(T1 | T2, G2) when
        # G1 ⊆ T2 ∪ G2 and T1 ⟂ (T2 ∪ G2) ∖ G1 | G1. E.g. under
        # Y1 ← X1 → X2 → Y2: P(Y1 | X1) P(Y2, X1) = P(Y1, Y2, X1).
        issubset(G1, union(T2, G2)) || return nothing
        E = collect(setdiff(union(T2, G2), G1))
        Ev = _rvsof(E)
        any(v -> v in T1v, Ev) && return nothing
        d_separated(g, T1v, Ev, _rvsof(collect(G1))) || return nothing
        new = _like(f1, vcat(f1.targets, f2.targets), f2.given)
        return (new, "$(_str(T1v)) ⟂ $(_str(Ev))$(_givenstr(_rvsof(collect(G1)))) (d-separation), chain rule: $(f1) $(f2) = ")
    end
    base = setdiff(G1, T2)
    if issubset(base, G2)
        E = collect(setdiff(G2, base))
        Ev = _rvsof(E)
        any(v -> v in T1v, Ev) && return nothing
        d_separated(g, T1v, Ev, _rvsof(collect(G1))) || return nothing
        new = _like(f1, vcat(f1.targets, f2.targets), f2.given)
        return (new, "$(_str(T1v)) ⟂ $(_str(Ev)) | $(_str(_rvsof(collect(G1)))) (d-separation), chain rule: $(f1) $(f2) = ")
    elseif issubset(G2, base)
        E = collect(setdiff(base, G2))
        Ev = _rvsof(E)
        any(v -> v in T2v, Ev) && return nothing
        d_separated(g, T2v, Ev, _rvsof(collect(G2))) || return nothing
        new = _like(f1, vcat(f1.targets, f2.targets), collect(base))
        return (new, "$(_str(T2v)) ⟂ $(_str(Ev))$(_givenstr(_rvsof(collect(G2)))) (d-separation), chain rule: $(f1) $(f2) = ")
    end
    return nothing
end

function rule_combine(p::Product, ctx::SimplifyContext)
    ctx.combine || return nothing
    fs = p.factors
    for i in eachindex(fs), j in eachindex(fs)
        i == j && continue
        f1, f2 = fs[i], fs[j]
        new = _chain_merge(f1, f2)
        new === nothing && continue
        rest = PExpr[fs[k] for k in eachindex(fs) if k != i && k != j]
        return (mul(new, rest...), "chain rule: $(f1) $(f2) = $(new)")
    end
    return nothing
end
rule_combine(::PExpr, ::SimplifyContext) = nothing

# Quotients: cancel common factors; P(A, B | C) / P(B | C) = P(A | B, C).
function rule_quotient(q::Quotient, ::SimplifyContext)
    q.num == q.den && return (Literal(1), "x / x = 1")
    nf = copy(_factors(q.num))
    df = copy(_factors(q.den))
    cancelled = false
    for f in copy(df)
        i = findfirst(==(f), nf)
        if i !== nothing
            deleteat!(nf, i)
            deleteat!(df, findfirst(==(f), df))
            cancelled = true
        end
    end
    cancelled && return (divide(mul(nf...), mul(df...)), "cancel common factors")
    if q.num isa Prob && q.den isa Prob && q.num.name === q.den.name
        n, d = q.num, q.den
        T, B = _atomset(n.targets), _atomset(d.targets)
        if issubset(B, T) && B != T && _atomset(n.given) == _atomset(d.given)
            new = _like(n, Atom[t for t in n.targets if !(t in B)], vcat(d.targets, n.given))
            return (new, "definition of conditional probability")
        end
    end
    return nothing
end
rule_quotient(::PExpr, ::SimplifyContext) = nothing

# Two terms of a sum that are the same except that one has P(S, E | C) where
# the other has P(S, Eᶜ | C) add up to the common part times P(S | C):
#   P(E | C) + P(Eᶜ | C) = 1,   P(A ∩ E) + P(A ∩ Eᶜ) = P(A).
function _complement_merge(f1, f2)
    (f1 isa Prob && f2 isa Prob && f1.name === f2.name) || return nothing
    _atomset(f1.given) == _atomset(f2.given) || return nothing
    d1 = Atom[t for t in f1.targets if !(t in f2.targets)]
    d2 = Atom[t for t in f2.targets if !(t in f1.targets)]
    (length(d1) == 1 && length(d2) == 1 && _iscomplementpair(d1[1], d2[1])) || return nothing
    rest = Atom[t for t in f1.targets if t != d1[1]]
    E = d1[1] isa Complement ? d2[1] : d1[1]
    new = isempty(rest) ? Literal(1) : _like(f1, rest, f1.given)
    why = isempty(rest) ? "complement rule: $(_like(f1, [E], f1.given)) + $(_like(f1, [complement(E)], f1.given)) = 1" :
          "sum over $(sprint(show, E)) and its complement"
    return (new, why)
end

function rule_sum(s::Sum, ::SimplifyContext)
    parts = map(_coef, s.terms)       # (coefficient, rest)
    for i in eachindex(parts), j in i+1:length(parts)
        (ci, ri), (cj, rj) = parts[i], parts[j]
        ci == cj || continue
        fi, fj = copy(_factors(ri)), copy(_factors(rj))
        # cancel the factors the two terms share
        common = PExpr[]
        for f in copy(fi)
            k = findfirst(==(f), fj)
            if k !== nothing
                push!(common, f)
                deleteat!(fj, k)
                deleteat!(fi, findfirst(==(f), fi))
            end
        end
        (length(fi) == 1 && length(fj) == 1) || continue
        m = _complement_merge(fi[1], fj[1])
        m === nothing && continue
        merged = mul(Literal(ci), m[1], common...)
        others = PExpr[s.terms[k] for k in eachindex(s.terms) if k != i && k != j]
        return (add(merged, others...), m[2])
    end
    # k − k P(E | C) = k P(Eᶜ | C)
    for i in eachindex(parts)
        k, r = parts[i]
        r == Literal(1) || continue
        for j in eachindex(parts)
            c, f = parts[j]
            (c == -k && f isa Prob && length(f.targets) == 1 && iseventlike(f.targets[1])) || continue
            new = mul(Literal(k), _like(f, Atom[complement(f.targets[1])], f.given))
            others = PExpr[s.terms[m] for m in eachindex(s.terms) if m != i && m != j]
            return (add(new, others...), "complement rule: 1 - $(f) = $(_like(f, Atom[complement(f.targets[1])], f.given))")
        end
    end
    return nothing
end
rule_sum(::PExpr, ::SimplifyContext) = nothing

# Event identities, one per step, for terms built with `@hold` (terms built
# normally are already canonical, so these never fire on them).

_compound(::Type{EventUnion}, ps) = isempty(ps) ? ∅ : length(ps) == 1 ? ps[1] : EventUnion(ps)
_compound(::Type{EventIntersection}, ps) = isempty(ps) ? Ω : length(ps) == 1 ? ps[1] : EventIntersection(ps)

# One identity at the root of an atom: `(new_atom, justification)` or `nothing`.
_eventstep(::Atom) = nothing
function _eventstep(c::Complement)
    e = c.event
    e isa Complement && return (e.event, "$(_astr(c)) = $(_astr(e.event))")
    e isa SureEvent && return (∅, "Ωᶜ = ∅")
    e isa ImpossibleEvent && return (Ω, "∅ᶜ = Ω")
    return nothing
end
function _eventstep(u::T) where {T<:Union{EventUnion,EventIntersection}}
    isunion = T === EventUnion
    unit, absorbing = isunion ? (ImpossibleEvent, SureEvent) : (SureEvent, ImpossibleEvent)
    op = isunion ? " ∪ " : " ∩ "
    ps = u.parts
    step(new) = (new, "$(_astr(u)) = $(_astr(new))")
    any(p -> p isa absorbing, ps) && return step(isunion ? Ω : ∅)
    any(p -> p isa unit, ps) && return step(_compound(T, Atom[p for p in ps if !(p isa unit)]))
    any(p -> p isa T, ps) && return step(_compound(T, reduce(vcat, (p isa T ? p.parts : Atom[p] for p in ps); init=Atom[])))
    for i in eachindex(ps), j in i+1:length(ps)
        a, b = ps[i], ps[j]
        if _iscomplementpair(a, b)
            new = isunion ? Ω : ∅
            return (new, "$(_astr(a, true))$op$(_astr(b, true)) = $(_astr(new))")
        elseif a == b
            return (_compound(T, Atom[p for (k, p) in enumerate(ps) if k != j]),
                    "$(_astr(a, true))$op$(_astr(a, true)) = $(_astr(a))")
        end
    end
    return nothing
end

# The first identity anywhere in an atom, outermost first.
function _eventstep_deep(a::Atom)
    s = _eventstep(a)
    s === nothing || return s
    if a isa Complement
        s = _eventstep_deep(a.event)
        s === nothing || return (Complement(s[1]), s[2])
    elseif a isa Union{EventUnion,EventIntersection}
        for (i, p) in enumerate(a.parts)
            s = _eventstep_deep(p)
            s === nothing && continue
            ps = copy(a.parts)
            ps[i] = s[1]
            return (typeof(a)(ps), s[2])
        end
    end
    return nothing
end

function rule_events(p::Prob, ::SimplifyContext)
    # identities inside the events
    for (side, xs) in ((:targets, p.targets), (:given, p.given)), i in eachindex(xs)
        s = _eventstep_deep(xs[i])
        s === nothing && continue
        t, g = copy(p.targets), copy(p.given)
        (side === :targets ? t : g)[i] = s[1]
        any(x -> x isa ImpossibleEvent, g) &&
            throw(ArgumentError("$(_astr(xs[i])) is the impossible event ∅: cannot condition on it"))
        return (_like(p, _heldflatten(t), _heldflatten(g)), s[2])
    end
    # identities of P itself: P(∅) = 0, contradictory targets, P(Ω) = 1, dropping Ω
    any(x -> x isa ImpossibleEvent, p.targets) && return (Literal(0), "P(∅) = 0")
    ts = _flatten_inter(p.targets)
    for i in eachindex(ts), j in i+1:length(ts)
        _iscomplementpair(ts[i], ts[j]) &&
            return (Literal(0), "$(_astr(ts[i], true)) ∩ $(_astr(ts[j], true)) = ∅, and P(∅) = 0")
    end
    if any(x -> x isa SureEvent, p.targets) || any(x -> x isa SureEvent, p.given)
        c = _mkprob(p.targets, p.given; name=p.name)
        return (c, c == Literal(1) ? "P(Ω) = 1" : "Ω always happens, so it can be dropped")
    end
    return nothing
end
rule_events(::PExpr, ::SimplifyContext) = nothing

# Later files (assume.jl, expectation.jl) push their rules onto this list.
const _RULES = Function[rule_events, rule_trivial, rule_independence, rule_docalculus, rule_marginal, rule_combine,
                rule_quotient, rule_sum]

# Try every rule at the root, then recurse into children (pre-order). Returns
# the rewritten expression and the justification of the one step taken.
function _rewrite_once(e::PExpr, ctx::SimplifyContext)
    for r in _RULES
        out = r(e, ctx)
        out === nothing || return out
    end
    cs = children(e)
    for (i, c) in enumerate(cs)
        out = _rewrite_once(c, ctx)
        if out !== nothing
            newcs = copy(cs)
            newcs[i] = out[1]
            return (rebuild(e, newcs), out[2])
        end
    end
    return nothing
end

"""
    simplify(expr; dag = nothing, assume = nothing, combine = true, trace = false, maxsteps = 200)
    simplify(expr, dag; kwargs...)
    simplify(expr, facts::Facts; kwargs...)

Rewrite `expr` to a simpler equal expression. Rules applied until none applies:

- the event identities (`A ∩ Aᶜ = ∅`, `(Aᶜ)ᶜ = A`, `A ∪ Ω = Ω`, `P(∅) = 0`,
  `P(Ω) = 1`, …), one at a time, for terms built with [`@hold`](@ref) (`∪`,
  `∩`, `'` and `P` otherwise apply them when the term is built);
- `P(A, B | A) = P(B | A)`, `P(A | A) = 1`, `P(Aᶜ | A) = 0`, `P(A ∪ B | A) = 1`;
- marginalisation, `∑_B P(A, B | C) = P(A | C)` and `∑_B P(B | C) = 1`, with
  factors free of the summed variables moved outside the sum;
- the chain rule backwards, `P(A | B, C) P(B | C) = P(A, B | C)` (`combine =
  false` turns this off);
- quotients: cancelling common factors and `P(A, B | C) / P(B | C) = P(A | B, C)`;
- sums: `P(E | C) + P(Eᶜ | C) = 1`, `P(A ∩ E) + P(A ∩ Eᶜ) = P(A)` (also with
  common factors), `1 - P(E) = P(Eᶜ)`;
- with a graph: conditional independence by d-separation, and do-calculus
  rules 1–3 for terms with `Do(⋅)`, and variable elimination that uses the
  graph to widen a factor's conditioning before the chain rule
  (`∑_B P(B | A) P(C | B) = P(C | A)` under `A → B → C`);
- with `assume` ([`assume`](@ref) facts): assumed independence drops
  conditioning and splits joints (`P(A ∩ B) = P(A) P(B)`), disjoint events
  have `P(A ∩ B) = 0` and `P(A ∪ B) = P(A) + P(B)`, and `implies(A, B)`
  gives `P(A ∩ B) = P(A)`;
- expectations: linearity, constants, `E[𝟙(A)] = P(A)`, taking out what is
  known, the tower property, and variances/covariances of constants and
  independent variables (see [`E`](@ref)).

A graph describes one distribution. `dag = g` (or `simplify(expr, g)`) applies
`g` to the default distribution `P` only; for named distributions pass a
`Dict`, e.g. `dag = Dict(:P => g, :q => gq)`. Terms of a distribution with no
graph get no independence rules, and terms of different distributions are
never combined.

With `trace = true` returns a [`Derivation`](@ref) recording each step and its
justification.
"""
function simplify(e::PExpr; dag=nothing, assume=nothing, combine::Bool=true,
                  trace::Bool=false, maxsteps::Integer=200)
    _held() && return _unheld(() -> simplify(e; dag, assume, combine, trace, maxsteps))
    ctx = SimplifyContext(_dags(dag), combine, _facts(assume))
    d = derivation(e)
    for _ in 1:maxsteps
        out = _rewrite_once(last(d), ctx)
        out === nothing && break
        _push!(d, out[1], out[2])
    end
    return trace ? d : last(d)
end
simplify(e::PExpr, g::Union{DAG,AbstractDict}; kwargs...) = simplify(e; dag=g, kwargs...)
simplify(e::PExpr, f::Facts; kwargs...) = simplify(e; assume=f, kwargs...)

# ---------------------------------------------------------------------------
# Expressing queries through a DAG's factorisation
# ---------------------------------------------------------------------------

function _check_in_graph(p::Prob, g::DAG)
    for v in _rvsof(vcat(p.targets, p.given))
        v in g.nodes || throw(ArgumentError("$(v.name) is not a node of the graph"))
    end
end

# The atoms of a query as substitutions on the graph's variables. Only atoms
# that pin a variable to one value can be substituted into the factorisation.
function _query_subs(p::Prob)
    subs = Pair{Atom,Atom}[]
    for a in Iterators.flatten((p.targets, p.given))
        t = a isa Intervention ? a.target : a
        if t isa Event
            push!(subs, t.rv => t)
        elseif t isa Complement && t.event isa RV && isevent(t.event)
            push!(subs, t.event => t)
        elseif t isa RV
            continue
        else
            throw(ArgumentError("expand/identify take variables, `X(value)` and named events or " *
                                "their complements; rewrite $(sprint(show, t)) first " *
                                "(inclusion_exclusion, demorgan)"))
        end
    end
    return subs
end

# Sum the factorisation over graph variables: ∑/∫ for ordinary variables,
# both truth values for event variables.
_sumvars(vs, body) = _sumover(Atom[v for v in vs], body)

"""
    expand(P(A | C), g::DAG; trace = false) -> PExpr

Write an observational query in terms of the graph's conditional
distributions `P(v | pa(v))`:

    P(A | C) = ∑_{V∖(A∪C)} ∏ P(v | pa(v)) / ∑_{V∖C} ∏ P(v | pa(v))

then simplify (without the chain rule backwards, so the result stays in terms
of the graph's own factors). Barren variables drop out, so this is symbolic
variable elimination. For event variables the sum is over the event and its
complement.
"""
function expand(p::Prob, g::DAG; trace::Bool=false)
    any(a -> a isa Intervention, p.given) && return identify(p, g; trace)
    _check_in_graph(p, g)
    subs = _query_subs(p)
    T = Set(_rvsof(p.targets))
    C = Set(_rvsof(p.given))
    F = factorize(g; name=p.name)
    num = _sumvars(RV[v for v in topological_order(g) if !(v in T) && !(v in C)], F)
    den = _sumvars(RV[v for v in topological_order(g) if !(v in C)], F)
    e = isempty(C) ? num : divide(num, den)
    e = substitute(e, subs...)
    d = derivation(p)
    _push!(d, e, "factorisation of the DAG")
    apply!(d, x -> simplify(x; dag=Dict(p.name => g), combine=false, trace=true))
    return trace ? d : last(d)
end

"""
    identify(P(Y | Do(X), W), g::DAG; trace = false) -> PExpr

An expression for an interventional query using only observational
probabilities, when every variable in `g` is observed. First tries the
do-calculus rules directly. If any `do(⋅)` remains, uses the truncated
factorisation (g-formula)

    P(V∖X | do(X)) = ∏_{v ∉ X} P(v | pa(v))

marginalised to `Y ∪ W` (and divided by its marginal over `W`), then simplifies.

With `latent = [U, …]` some variables are hidden: the graph is projected to
an ADMG ([`latent_projection`](@ref)) and the ID/IDC algorithm decides the
query (see `identify(::Prob, ::ADMG)`).
"""
function identify(p::Prob, g::DAG; trace::Bool=false, latent=nothing)
    latent === nothing || return identify(p, latent_projection(g, latent); trace)
    _hascf(p) && return identify(p, latent_projection(g, RV[]); trace)     # counterfactuals: counterfactual.jl
    _check_in_graph(p, g)
    d = simplify(p; dag=Dict(p.name => g), combine=false, trace=true)
    q = last(d)
    if q isa Prob && any(a -> a isa Intervention, q.given)
        subs = _query_subs(q)
        X = Set(_rvsof(a for a in q.given if a isa Intervention))
        Y = Set(_rvsof(q.targets))
        W = Set(_rvsof(a for a in q.given if !(a isa Intervention)))
        F = mul((Prob(Atom[v], Atom[pp for pp in g.parents[v]], q.name)
                 for v in topological_order(g) if !(v in X))...)
        num = _sumvars(RV[v for v in topological_order(g) if !(v in X) && !(v in Y) && !(v in W)], F)
        e = isempty(W) ? num :
            divide(num, _sumvars(RV[v for v in topological_order(g) if !(v in X) && !(v in W)], F))
        e = substitute(e, subs...)
        _push!(d, e, "truncated factorisation (g-formula)")
        apply!(d, x -> simplify(x; dag=Dict(p.name => g), combine=false, trace=true))
    end
    return trace ? d : last(d)
end

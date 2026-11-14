# Rules of the algebra of events: complements, De Morgan, inclusion–exclusion.
# (Total probability is `condition_on`/`total_probability` in rules.jl.)

"""
    demorgan(x)

Apply De Morgan's laws everywhere in an event or expression:
`(A ∪ B)ᶜ = Aᶜ ∩ Bᶜ` and `(A ∩ B)ᶜ = Aᶜ ∪ Bᶜ`, pushing complements down to
the named events.
"""
demorgan(a::Atom) = a
demorgan(e::EventUnion) = _evunion(Atom[demorgan(x) for x in e.parts])
demorgan(e::EventIntersection) = _evintersect(Atom[demorgan(x) for x in e.parts])
function demorgan(c::Complement)
    e = c.event
    e isa EventUnion && return _evintersect(Atom[demorgan(complement(x)) for x in e.parts])
    e isa EventIntersection && return _evunion(Atom[demorgan(complement(x)) for x in e.parts])
    return c
end
demorgan(i::Intervention) = i
demorgan(e::PExpr) = mapprobs(p -> _mkprob(map(demorgan, p.targets), map(demorgan, p.given); name=p.name), e)

"""
    complement_rule(P(E | C)) -> PExpr

`P(E | C) = 1 - P(Eᶜ | C)`. With several targets, `E` is their intersection.
"""
function complement_rule(p::Prob)
    all(iseventlike, p.targets) ||
        throw(ArgumentError("the complement rule needs events; $(p) has a plain variable as a target"))
    E = _evintersect(p.targets)
    return add(Literal(1), -_mkprob(Atom[complement(E)], p.given; name=p.name))
end

"""
    inclusion_exclusion(P(A ∪ B ∪ ⋯ | C)) -> PExpr

The inclusion–exclusion principle:
`P(A ∪ B | C) = P(A | C) + P(B | C) - P(A ∩ B | C)`, and in general the
alternating sum over every intersection of the parts.

If the target is a complement of a union, or several targets intersect a
union, it uses `P(Eᶜ) = 1 - P(E)` and `P(X ∩ (A ∪ B)) = P(X ∩ A) + P(X ∩ B) -
P(X ∩ A ∩ B)` as needed.
"""
function inclusion_exclusion(p::Prob)
    ui = findfirst(t -> t isa EventUnion || (t isa Complement && t.event isa EventUnion), p.targets)
    ui === nothing && throw(ArgumentError("$(p) has no union among its targets"))
    u = p.targets[ui]
    others = Atom[t for (k, t) in enumerate(p.targets) if k != ui]
    if u isa Complement      # P(X ∩ (A ∪ B)ᶜ) = P(X) - P(X ∩ (A ∪ B))
        whole = isempty(others) ? Literal(1) : _mkprob(others, p.given; name=p.name)
        return add(whole, -inclusion_exclusion(_like(p, vcat(others, u.event), p.given)))
    end
    parts = u.parts
    terms = PExpr[]
    for k in 1:length(parts)
        for S in _combinations(parts, k)
            push!(terms, mul(Literal(isodd(k) ? 1 : -1), _mkprob(vcat(others, S), p.given; name=p.name)))
        end
    end
    return add(terms...)
end

"""
    union_bound(P(A ∪ B ∪ ⋯ | C)) -> PExpr

Boole's inequality's right-hand side, `P(A | C) + P(B | C) + ⋯`: an upper bound
on the probability of the union, not an equality.
"""
function union_bound(p::Prob)
    length(p.targets) == 1 && p.targets[1] isa EventUnion ||
        throw(ArgumentError("union_bound needs a single union as the target"))
    return add((_like(p, Atom[x], p.given) for x in p.targets[1].parts)...)
end

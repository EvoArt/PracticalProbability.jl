# Information theory: entropy, mutual information and divergences.
#
# All of these are expectations of logs of probability terms, so they are
# probability expressions (`PExpr`) like `E[⋯]`: they go through `simplify`,
# `latex` and `compute`, and `definition` writes them out as expectations.
#
# Conventions (as in Cover & Thomas):
#
# * `H(X | Z)` and `MI(X, Y | Z)` average over the plain variables they
#   condition on, so they are numbers. Conditioning on an event (`Z(1)`, a
#   named event) fixes it, and `Do(X)` with a free `X` leaves a function of `x`.
# * A divergence `KL(p(X | Z) ‖ q(X | Z))` is pointwise: a function of `z`, like
#   `E[Y | Z]`. The averaged conditional divergence of the KL chain rule is
#   `E(KL(⋯))` over `p(z)`, which `chainrule` writes out.

# ---------------------------------------------------------------------------
# Terms
# ---------------------------------------------------------------------------

"""
    Entropy(X, given, name)

`H(X | given)` under the distribution `name`. Build with [`H`](@ref).
"""
struct Entropy <: PExpr
    X::Vector{Atom}
    given::Vector{Atom}
    name::Symbol
end

"""
    MutualInformation(X, Y, given, name)

`I(X; Y | given)`. Build with [`MI`](@ref).
"""
struct MutualInformation <: PExpr
    X::Vector{Atom}
    Y::Vector{Atom}
    given::Vector{Atom}
    name::Symbol
end

"""
    Divergence(p, q, kind, α)

A divergence between two distributions of the same variables: `kind` is
`:kl` ([`KL`](@ref)), `:cross` ([`crossentropy`](@ref)), `:renyi`
([`renyi`](@ref), order `α`) or `:chisq` ([`chisq`](@ref)).
"""
struct Divergence <: PExpr
    p::Prob
    q::Prob
    kind::Symbol
    α::Float64
end

"""
    Log(e)

`log e` for a probability expression. Build with `log(e)`.
"""
struct Log <: PExpr
    arg::PExpr
end

_isvarlike(a::Atom) = a isa RV || a isa IndexedRV
function _check_vars(xs, what)
    all(_isvarlike, xs) || throw(ArgumentError(
        "$what takes random variables before the bar (not events such as X(1)); got $(_atomlist(xs))"))
    return xs
end

function _parse_info(args, name)
    p = P(args...; name)
    p isa Prob || throw(ArgumentError("H(⋯) needs at least one variable"))
    return p
end

"""
    H(X)
    H(X, Y | Z; name = :P)
    ℍ(…)

The (joint, conditional) entropy, in nats. Conditioning on a variable averages
over it (`H(X | Z) = ∑_z P(z) H(X | Z = z)`); conditioning on an event such as
`Z(1)` fixes it. `name` picks the distribution (`H(X; name = :q)` is `H_q(X)`).

Rules (in [`simplify`](@ref)): `H(X | X, Z) = 0`, conditioning that is
independent of `X` (by a graph or assumed facts) drops out, the entropy of
independent parts splits, and sums of entropies recombine by the chain rule
(`H(X | Z) - H(X | Y, Z) = I(X; Y | Z)`). Explicit: [`chainrule`](@ref),
[`definition`](@ref) (`E[-log P(X | Z)]`), [`entropies`](@ref).

`H(f::Family)` is the closed-form entropy of a distribution family.
"""
function H(args...; name::Symbol=:P)
    p = _parse_info(args, name)
    return Entropy(_check_vars(p.targets, "H"), p.given, name)
end
const ℍ = H

"""
    MI(X, Y)
    MI(X, Y | Z; name = :P)
    MI((X₁, X₂), Y | Z)

The (conditional) mutual information `I(X; Y | Z)`, in nats. Julia reads `;`
inside a call as the start of keyword arguments, so the two sides are
separated by a comma.

Rules: `I(X; Y | Z) = 0` when `X ⟂ Y | Z` (by a graph or facts),
`I(X; X | Z) = H(X | Z)`, and `I(X; Y | Z) = 0` when `X` is known given `Z`.
Explicit: [`chainrule`](@ref), [`definition`](@ref), [`entropies`](@ref).
"""
function MI(a, b, rest...; name::Symbol=:P)
    X = _check_vars(a isa Atom ? Atom[a] : _atoms(a), "MI")
    given = Atom[]
    if b isa Given
        Y = b.targets
        append!(given, b.given)
    else
        Y = b isa Atom ? Atom[b] : _atoms(b)
    end
    _check_vars(Y, "MI")
    for r in rest
        append!(given, r isa Atom ? Atom[r] : _atoms(r))
    end
    return MutualInformation(unique(X), unique(Y), unique(_flatten_inter(given)), name)
end

function _divergence(p::PExpr, q::PExpr, kind::Symbol, α::Real=NaN)
    (p isa Prob && q isa Prob) ||
        throw(ArgumentError("a divergence compares two probability terms, e.g. KL(P(X | Z), q(X | Z))"))
    _check_vars(p.targets, "a divergence")
    issetequal(p.targets, q.targets) ||
        throw(ArgumentError("both distributions must be over the same variables: $(p) and $(q)"))
    return Divergence(p, q, kind, Float64(α))
end

"""
    KL(p, q)
    KL(P(X | Z), q(X | Z))

The Kullback–Leibler divergence `D_KL(p ‖ q) = E_p[log p - log q]`, in nats.
It is pointwise in the conditioning variables (a function of `z`); average it
with `E(KL(⋯))`. `KL(p, p) = 0`; [`chainrule`](@ref) splits a joint
divergence; [`definition`](@ref) writes the expectation out.

`KL(f::Family, g::Family)` is the closed form for distribution families.
"""
KL(p::PExpr, q::PExpr) = _divergence(p, q, :kl)

"""
    crossentropy(p, q)

The cross-entropy `H(p, q) = E_p[-log q] = H(p) + KL(p ‖ q)` (pointwise in the
conditioning variables).
"""
crossentropy(p::PExpr, q::PExpr) = _divergence(p, q, :cross)

"""
    renyi(p, q, α)

The Rényi divergence of order `α`,
`D_α(p ‖ q) = log E_p[(p / q)^(α - 1)] / (α - 1)`. `D_1` is `KL`, and
`D_2 = log(1 + χ²(p ‖ q))`.
"""
function renyi(p::PExpr, q::PExpr, α::Real)
    α > 0 || throw(ArgumentError("the order of a Rényi divergence must be positive"))
    α == 1 && return KL(p, q)
    return _divergence(p, q, :renyi, α)
end

"""
    chisq(p, q)

The χ² divergence `χ²(p ‖ q) = E_q[(p / q)²] - 1 = E_p[p / q] - 1`. For
importance sampling from `q` towards `p`, the relative effective sample size is
`1 / (1 + χ²(p ‖ q))` (see [`relative_ess`](@ref)).
"""
chisq(p::PExpr, q::PExpr) = _divergence(p, q, :chisq)

# log of an expression, with the arithmetic of constants done on the way
function _log(e::PExpr)
    e isa Literal && return Literal(log(e.value))
    return Log(e)
end
Base.log(e::PExpr) = _log(e)

# ---------------------------------------------------------------------------
# Equality, variables, traversal
# ---------------------------------------------------------------------------

Base.:(==)(a::Entropy, b::Entropy) = a.name === b.name && issetequal(a.X, b.X) && issetequal(a.given, b.given)
Base.hash(a::Entropy, h::UInt) = hash(a.name, hash(Set(a.given), hash(Set(a.X), hash(:Entropy, h))))
function Base.:(==)(a::MutualInformation, b::MutualInformation)
    a.name === b.name && issetequal(a.given, b.given) || return false
    return (issetequal(a.X, b.X) && issetequal(a.Y, b.Y)) || (issetequal(a.X, b.Y) && issetequal(a.Y, b.X))
end
Base.hash(a::MutualInformation, h::UInt) =
    hash(a.name, hash(Set(a.given), hash(hash(Set(a.X)) + hash(Set(a.Y)), hash(:MutualInformation, h))))
Base.:(==)(a::Divergence, b::Divergence) =
    a.kind === b.kind && isequal(a.α, b.α) && a.p == b.p && a.q == b.q
Base.hash(a::Divergence, h::UInt) = hash(a.q, hash(a.p, hash(a.α, hash(a.kind, hash(:Divergence, h)))))
Base.:(==)(a::Log, b::Log) = a.arg == b.arg
Base.hash(a::Log, h::UInt) = hash(a.arg, hash(:Log, h))

const _InfoTerm = Union{Entropy,MutualInformation}

# Free variables: what the quantity is a function of. Entropies average over
# their plain conditioning variables, so only free do(⋅) variables remain.
_dofree(given) = Set{RV}(v for v in (freerv(a) for a in given if a isa Intervention) if v !== nothing)
_freevars(h::_InfoTerm) = _dofree(h.given)
_freevars(d::Divergence) = union(_freevars(d.p), _freevars(d.q)) |> s -> setdiff(s, Set(_rvsof(d.p.targets)))
_freevars(l::Log) = _freevars(l.arg)
_allvars(h::Entropy) = Set{RV}(_rvsof(vcat(h.X, h.given)))
_allvars(m::MutualInformation) = Set{RV}(_rvsof(vcat(m.X, m.Y, m.given)))
_allvars(d::Divergence) = union(_allvars(d.p), _allvars(d.q))
_allvars(l::Log) = _allvars(l.arg)

children(::Union{_InfoTerm,Divergence}) = PExpr[]
rebuild(e::Union{_InfoTerm,Divergence}, cs) = e
children(l::Log) = PExpr[l.arg]
rebuild(::Log, cs) = _log(cs[1])

_subs(h::Entropy, d) = Entropy(Atom[_sa(a, d) for a in h.X], Atom[_sa(a, d) for a in h.given], h.name)
_subs(m::MutualInformation, d) = MutualInformation(Atom[_sa(a, d) for a in m.X], Atom[_sa(a, d) for a in m.Y],
                                                   Atom[_sa(a, d) for a in m.given], m.name)
function _subs(dv::Divergence, d)
    p, q = _subs(dv.p, d), _subs(dv.q, d)
    (p isa Prob && q isa Prob) || throw(ArgumentError("substitution turned $(dv) into a degenerate divergence"))
    return Divergence(p::Prob, q::Prob, dv.kind, dv.α)
end
_subs(l::Log, d) = _log(_subs(l.arg, d))

_atomic(::Union{_InfoTerm,Divergence,Log}) = true

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_hletter(name::Symbol, base) = name === :P ? base : string(base, "_", name)

Base.show(io::IO, h::Entropy) = print(io, _hletter(h.name, "H"), "(", _atomlist(h.X), _cond(h.given), ")")
Base.show(io::IO, m::MutualInformation) =
    print(io, _hletter(m.name, "I"), "(", _atomlist(m.X), "; ", _atomlist(m.Y), _cond(m.given), ")")
function Base.show(io::IO, d::Divergence)
    if d.kind === :cross
        print(io, "H(", d.p, ", ", d.q, ")")
    else
        head = d.kind === :kl ? "KL" : d.kind === :chisq ? "χ²" : string("D_", _fmtα(d.α))
        print(io, head, "(", d.p, " ‖ ", d.q, ")")
    end
end
Base.show(io::IO, l::Log) = print(io, "log(", l.arg, ")")

_fmtα(α) = isinteger(α) ? string(Int(α)) : string(α)

_lxhead(name::Symbol, base) = name === :P ? base : string(base, "_{", _lx(name), "}")
_latex(h::Entropy) = string(_lxhead(h.name, "H"), "\\left(", _lxatomlist(h.X), _lxcond(h.given), "\\right)")
_latex(m::MutualInformation) = string(_lxhead(m.name, "I"), "\\left(", _lxatomlist(m.X), "; ",
                                      _lxatomlist(m.Y), _lxcond(m.given), "\\right)")
function _latex(d::Divergence)
    d.kind === :cross && return string("H\\left(", _latex(d.p), ", ", _latex(d.q), "\\right)")
    head = d.kind === :kl ? "D_{\\mathrm{KL}}" : d.kind === :chisq ? "\\chi^{2}" : string("D_{", _fmtα(d.α), "}")
    return string(head, "\\left(", _latex(d.p), " \\,\\middle\\|\\, ", _latex(d.q), "\\right)")
end
_latex(l::Log) = l.arg isa Prob ? string("\\log ", _latex(l.arg)) : string("\\log\\left(", _latex(l.arg), "\\right)")

# ---------------------------------------------------------------------------
# Explicit transformations
# ---------------------------------------------------------------------------

_fixed(given) = Atom[a for a in given if a isa Intervention || freerv(a) === nothing]
_logq(e::PExpr) = RFun(:log, RQ(e))

"""
    definition(H(X | Z))      -> E[-log P(X | Z)]
    definition(MI(X, Y | Z))  -> E[log P(X, Y | Z) / (P(X | Z) P(Y | Z))]
    definition(KL(p, q))      -> E_p[log p - log q]
    definition(crossentropy(p, q)) -> E_p[-log q]
    definition(chisq(p, q))   -> E_p[p / q] - 1
    definition(renyi(p, q, α)) -> log E_p[(p / q)^(α - 1)] / (α - 1)
    definition(log(…))         unchanged

Information quantities as expectations of logs. Conditioning on plain variables
is averaged over (the expectation is over them too); events and `do(⋅)` stay
conditioned on.
"""
definition(h::Entropy) = Expect(rmul(RConst(-1), _logq(Prob(h.X, h.given, h.name))), _fixed(h.given), h.name)
function definition(m::MutualInformation)
    r = divide(Prob(vcat(m.X, m.Y), m.given, m.name),
               mul(Prob(m.X, m.given, m.name), Prob(m.Y, m.given, m.name)))
    return Expect(_logq(r), _fixed(m.given), m.name)
end
function definition(d::Divergence)
    p, q = d.p, d.q
    d.kind === :kl && return Expect(radd(_logq(p), rmul(RConst(-1), _logq(q))), p.given, p.name)
    d.kind === :cross && return Expect(rmul(RConst(-1), _logq(q)), p.given, p.name)
    d.kind === :chisq && return add(Expect(RQ(divide(p, q)), p.given, p.name), Literal(-1))
    return mul(Literal(1 / (d.α - 1)), _log(Expect(rpow(RQ(divide(p, q)), d.α - 1), p.given, p.name)))
end

_ordered(order, xs) = _aslist(order === nothing ? xs : order)

"""
    chainrule(H(X₁, …, Xₙ | Z); order)   -> ∑ᵢ H(Xᵢ | Xᵢ₊₁, …, Xₙ, Z)
    chainrule(MI((X₁, …, Xₙ), Y | Z); order) -> ∑ᵢ I(Xᵢ; Y | Xᵢ₊₁, …, Xₙ, Z)
    chainrule(KL(p(X₁, …, Xₙ | Z), q(⋯)); order)
        -> ∑ᵢ E_p[KL(p(Xᵢ | Xᵢ₊₁, …, Z) ‖ q(Xᵢ | Xᵢ₊₁, …, Z)) | Z]

The chain rules of information theory, peeling variables off in `order` (the
same order as `chainrule(P(⋯))`).
"""
function chainrule(h::Entropy; order=nothing)
    o = _ordered(order, h.X)
    issetequal(o, h.X) || throw(ArgumentError("order must list exactly the variables of $(h)"))
    return add((Entropy(Atom[o[i]], vcat(o[i+1:end], h.given), h.name) for i in eachindex(o))...)
end
function chainrule(m::MutualInformation; order=nothing)
    o = _ordered(order, m.X)
    issetequal(o, m.X) || throw(ArgumentError("order must list exactly the first variables of $(m)"))
    return add((MutualInformation(Atom[o[i]], m.Y, vcat(o[i+1:end], m.given), m.name) for i in eachindex(o))...)
end
function chainrule(d::Divergence; order=nothing)
    d.kind === :kl || throw(ArgumentError("the chain rule is for KL divergences"))
    p, q = d.p, d.q
    o = _ordered(order, p.targets)
    issetequal(o, p.targets) || throw(ArgumentError("order must list exactly the variables of $(p)"))
    terms = PExpr[]
    for i in eachindex(o)
        rest = o[i+1:end]
        k = Divergence(_like(p, Atom[o[i]], vcat(rest, p.given)), _like(q, Atom[o[i]], vcat(rest, q.given)), :kl, NaN)
        push!(terms, isempty(rest) ? k : Expect(RQ(k), p.given, p.name))
    end
    return add(terms...)
end

"""
    entropies(expr; joint = false) -> PExpr

Rewrite every mutual information as entropies,
`I(X; Y | Z) = H(X | Z) + H(Y | Z) - H(X, Y | Z)`. With `joint = true`, also
every conditional entropy as a difference of joint ones,
`H(X | Z) = H(X, Z) - H(Z)` (the form the inequality prover works in).
"""
entropies(e::PExpr; joint::Bool=false) = _entropies(e, joint)
_entropies(e::PExpr, joint) = rebuild(e, PExpr[_entropies(c, joint) for c in children(e)])
_entropies(e::Union{Prob,Literal}, joint) = e
function _entropies(m::MutualInformation, joint)
    G, n = m.given, m.name
    out = add(Entropy(m.X, G, n), Entropy(m.Y, G, n), -Entropy(unique(vcat(m.X, m.Y)), G, n))
    return joint ? _entropies(out, true) : out
end
function _entropies(h::Entropy, joint)
    (joint && !isempty(h.given)) || return h
    all(a -> _isvarlike(a), h.given) ||
        throw(ArgumentError("$(h) conditions on an event or do(⋅), so it has no joint-entropy form"))
    return add(Entropy(unique(vcat(h.X, h.given)), Atom[], h.name), -Entropy(h.given, Atom[], h.name))
end

"""
    information(s::Independence; name = :P) -> MutualInformation

The mutual information that the independence statement says is zero:
`X ⟂ Y | Z` is `I(X; Y | Z) = 0`.
"""
information(s::Independence; name::Symbol=:P) =
    MutualInformation(Atom[s.X...], Atom[s.Y...], Atom[s.Z...], name)

# ---------------------------------------------------------------------------
# Inequalities
# ---------------------------------------------------------------------------

"""
    Inequality(lhs, rhs, why = "")
    lhs ≤ rhs

The statement `lhs ≤ rhs` between probability expressions, e.g.
`MI(X, Z) ≤ MI(X, Y)`. Returned by [`data_processing`](@ref), checked
numerically by [`verify`](@ref) and proved (when it is a Shannon-type
inequality) by [`prove`](@ref). `a ≥ b` is `b ≤ a`.
"""
struct Inequality
    lhs::PExpr
    rhs::PExpr
    why::String
end
Inequality(lhs::PExpr, rhs::PExpr) = Inequality(lhs, rhs, "")
_pe(x::PExpr) = x
_pe(x::Number) = Literal(x)
Base.:(<=)(a::PExpr, b::Union{PExpr,Number}) = Inequality(a, _pe(b))
Base.:(<=)(a::Number, b::PExpr) = Inequality(Literal(a), b)
Base.:(>=)(a::PExpr, b::Union{PExpr,Number}) = Inequality(_pe(b), a)
Base.:(>=)(a::Number, b::PExpr) = Inequality(b, Literal(a))

Base.:(==)(a::Inequality, b::Inequality) = a.lhs == b.lhs && a.rhs == b.rhs
function Base.show(io::IO, q::Inequality)
    show(io, q.lhs)
    print(io, " ≤ ")
    show(io, q.rhs)
end
function Base.show(io::IO, ::MIME"text/plain", q::Inequality)
    show(io, q)
    isempty(q.why) || print(io, "    (", q.why, ")")
end
_latex(q::Inequality) = string(_latex(q.lhs), " \\le ", _latex(q.rhs))
Base.show(io::IO, ::MIME"text/latex", q::Inequality) = print(io, "\$\$", _latex(q), "\$\$")

"""
    verify(ineq::Inequality, nets; atol = 1e-10) -> Bool

Check `lhs ≤ rhs` numerically (everywhere, if either side is a function).
"""
function verify(q::Inequality, nets::_Nets; atol::Real=1e-10)
    a, b = _asfactor(compute(q.lhs, nets)), _asfactor(compute(q.rhs, nets))
    vars = union(a.vars, b.vars)
    return all(broadcast((x, y) -> x <= y + atol, _align(a, vars), _align(b, vars)))
end

"""
    data_processing(g::DAG, X, Y, Z; name = :P) -> Inequality

The data-processing inequality `I(X; Z) ≤ I(X; Y)`, which holds when the graph
makes `X → Y → Z` a Markov chain (`X ⟂ Z | Y` by d-separation). Throws if the
graph does not imply it.
"""
function data_processing(g::DAG, X, Y, Z; name::Symbol=:P)
    Xs, Ys, Zs = _rvlist(X), _rvlist(Y), _rvlist(Z)
    d_separated(g, Xs, Zs, Ys) || throw(ArgumentError(
        "the graph does not make $(_str(Xs)) → $(_str(Ys)) → $(_str(Zs)) a Markov chain " *
        "($(_str(Xs)) and $(_str(Zs)) are not d-separated by $(_str(Ys)))"))
    return Inequality(MutualInformation(Atom[Xs...], Atom[Zs...], Atom[], name),
                      MutualInformation(Atom[Xs...], Atom[Ys...], Atom[], name),
                      "data processing: $(_str(Xs)) ⟂ $(_str(Zs)) | $(_str(Ys))")
end

# ---------------------------------------------------------------------------
# Variational inference
# ---------------------------------------------------------------------------

"""
    elbo(joint, q) -> Expect

The evidence lower bound `E_q[log p(y, z) - log q(z | y) | y]` for a joint
`p(y, z)` (e.g. `P(y, z)`) and a variational distribution `q(z | y)`. The
latent variables `z` are the targets of `q`.
"""
function elbo(joint::Prob, q::Prob)
    Z = _rvsof(q.targets)
    all(v -> v in _rvsof(joint.targets), Z) ||
        throw(ArgumentError("the variational distribution $(q) must be over latent variables of $(joint)"))
    return Expect(radd(_logq(joint), rmul(RConst(-1), _logq(q))), q.given, q.name)
end

"""
    evidence_decomposition(joint, q) -> Derivation

The identity `log p(y) = ELBO + KL(q(z | y) ‖ p(z | y))`, derived step by step
for the joint `p(y, z)` and variational distribution `q(z | y)`. Since the KL
divergence is non-negative, the ELBO is a lower bound on the log evidence.
`Equation(d)` gives the identity; `verify(d, nets)` checks every line.
"""
function evidence_decomposition(joint::Prob, q::Prob)
    Z = Atom[q.targets...]
    Y = Atom[a for a in joint.targets if !(a in Z)]
    py = _like(joint, Y, joint.given)
    post = _like(joint, Z, vcat(Y, joint.given))
    d = derivation(_log(py))
    _push!(d, Expect(_logq(py), q.given, q.name), "the expectation of a constant (log p(y) does not depend on z)")
    _push!(d, Expect(_logq(divide(joint, post)), q.given, q.name), "p(y) = p(y, z) / p(z | y)")
    _push!(d, Expect(radd(_logq(divide(joint, q)), _logq(divide(q, post))), q.given, q.name),
           "multiply and divide by q(z | y) inside the log")
    _push!(d, add(elbo(joint, q), Divergence(q, post, :kl, NaN)), "ELBO + KL(q ‖ posterior)")
    return d
end

# ---------------------------------------------------------------------------
# Simplification rules
# ---------------------------------------------------------------------------

_hasdo(given) = any(a -> a isa Intervention, given)

function rule_entropy(h::Entropy, ctx::SimplifyContext)
    isempty(h.X) && return (Literal(0), "the entropy of nothing is 0")
    gv = _rvsof(h.given)
    keep = Atom[x for x in h.X if !(x in h.given)]
    isempty(keep) && return (Literal(0), "a known variable has no entropy: H(X | X, ⋯) = 0")
    length(keep) < length(h.X) &&
        return (Entropy(keep, h.given, h.name), "drop variables that are conditioned on")
    _hasdo(h.given) && return nothing
    T = _rvsof(h.X)
    # drop conditioning that is independent of the targets
    for w in h.given
        rest = _rvsof(a for a in h.given if a != w)
        wv = rvs(w)
        (isempty(wv) || any(v -> v in T || v in rest, wv)) && continue
        why = _independent(ctx, h.name, T, wv, rest)
        why === nothing && continue
        return (Entropy(h.X, Atom[a for a in h.given if a != w], h.name), why)
    end
    # the entropy of independent parts adds up
    if length(h.X) > 1
        for k in 1:length(h.X)-1, part in _combinations(h.X, k)
            first(part) == first(h.X) || continue        # each split once
            other = Atom[x for x in h.X if !(x in part)]
            why = _independent(ctx, h.name, _rvsof(part), _rvsof(other), gv)
            why === nothing && continue
            return (add(Entropy(part, h.given, h.name), Entropy(other, h.given, h.name)),
                    "the entropy of independent parts adds up: " * why)
        end
    end
    return nothing
end
rule_entropy(::PExpr, ::SimplifyContext) = nothing

function rule_mutualinfo(m::MutualInformation, ctx::SimplifyContext)
    (isempty(m.X) || isempty(m.Y)) && return (Literal(0), "information with nothing is 0")
    issetequal(m.X, m.Y) && return (Entropy(m.X, m.given, m.name), "I(X; X | Z) = H(X | Z)")
    for (side, other, first_) in ((m.X, m.Y, true), (m.Y, m.X, false))
        keep = Atom[x for x in side if !(x in m.given)]
        isempty(keep) && return (Literal(0), "a known variable carries no information: I(X; Y | X, ⋯) = 0")
        if length(keep) < length(side)
            return (MutualInformation(first_ ? keep : other, first_ ? other : keep, m.given, m.name),
                    "drop variables that are conditioned on")
        end
    end
    _hasdo(m.given) && return nothing
    X, Y, G = _rvsof(m.X), _rvsof(m.Y), _rvsof(m.given)
    isempty(intersect(X, Y)) || return nothing
    why = _independent(ctx, m.name, X, Y, G)
    why === nothing || return (Literal(0), "independent, so no mutual information: " * why)
    for w in m.given
        rest = _rvsof(a for a in m.given if a != w)
        wv = rvs(w)
        (isempty(wv) || any(v -> v in X || v in Y || v in rest, wv)) && continue
        why = _independent(ctx, m.name, vcat(X, Y), wv, rest)
        why === nothing && continue
        return (MutualInformation(m.X, m.Y, Atom[a for a in m.given if a != w], m.name), why)
    end
    return nothing
end
rule_mutualinfo(::PExpr, ::SimplifyContext) = nothing

function rule_divergence(d::Divergence, ::SimplifyContext)
    if d.p == d.q
        d.kind === :cross || return (Literal(0), "a distribution has zero divergence from itself")
        isempty(_freevars(d)) &&
            return (Entropy(d.p.targets, d.p.given, d.p.name), "the cross-entropy of p with itself is its entropy")
    end
    return nothing
end
rule_divergence(::PExpr, ::SimplifyContext) = nothing

function rule_log(l::Log, ::SimplifyContext)
    a = l.arg
    a isa Literal && return (Literal(log(a.value)), "arithmetic")
    if a isa Product
        return (add((_log(f) for f in a.factors)...), "the log of a product is the sum of the logs")
    elseif a isa Quotient
        return (add(_log(a.num), -_log(a.den)), "the log of a quotient is the difference of the logs")
    elseif a isa Power
        return (mul(Literal(a.p), _log(a.base)), "log xᵖ = p log x")
    end
    return nothing
end
rule_log(::PExpr, ::SimplifyContext) = nothing

# Sums of information terms: the chain rules read backwards.
_aset(xs) = Set{Atom}(xs)
function _info_pair(a::PExpr, b::PExpr, same::Bool)
    # a and b are the (coefficient-free) parts of two terms; `same` says
    # whether their coefficients are equal (true) or opposite (false).
    if a isa Entropy && b isa Entropy && a.name === b.name
        A, GA, B, GB = _aset(a.X), _aset(a.given), _aset(b.X), _aset(b.given)
        if !same
            # H(A | G) - H(A | G, B) = I(A; B | G)
            if A == B && issubset(GA, GB) && GA != GB
                extra = setdiff(GB, GA)
                isempty(intersect(extra, A)) && all(_isvarlike, extra) &&
                    return (MutualInformation(a.X, Atom[x for x in b.given if x in extra], a.given, a.name),
                            "H(A | G) - H(A | B, G) = I(A; B | G)", 1)
            end
            # H(A, B | G) - H(B | G) = H(A | B, G)
            if GA == GB && issubset(B, A) && A != B
                return (Entropy(Atom[x for x in a.X if !(x in B)], vcat(b.X, a.given), a.name),
                        "chain rule: H(A, B | G) - H(B | G) = H(A | B, G)", 1)
            end
        else
            # H(A | G) + H(B | A, G) = H(A, B | G)
            if GB == union(GA, A) && isempty(intersect(B, GB))
                return (Entropy(vcat(a.X, b.X), a.given, a.name), "chain rule: H(A | G) + H(B | A, G) = H(A, B | G)", 1)
            end
        end
    elseif !same && a isa Entropy && b isa MutualInformation && a.name === b.name &&
           _aset(a.given) == _aset(b.given)
        # H(A | G) - I(A; B | G) = H(A | B, G)
        for (s, o) in ((b.X, b.Y), (b.Y, b.X))
            _aset(s) == _aset(a.X) || continue
            isempty(intersect(_aset(o), _aset(a.given))) || continue
            return (Entropy(a.X, vcat(o, a.given), a.name), "H(A | G) - I(A; B | G) = H(A | B, G)", 1)
        end
    elseif same && a isa MutualInformation && b isa MutualInformation && a.name === b.name
        # I(A; B | G) + I(A; C | B, G) = I(A; B, C | G)
        for (A1, B1) in ((a.X, a.Y), (a.Y, a.X)), (A2, C2) in ((b.X, b.Y), (b.Y, b.X))
            _aset(A1) == _aset(A2) || continue
            _aset(b.given) == union(_aset(a.given), _aset(B1)) || continue
            isempty(intersect(_aset(C2), _aset(b.given))) || continue
            return (MutualInformation(A1, vcat(B1, C2), a.given, a.name),
                    "chain rule: I(A; B | G) + I(A; C | B, G) = I(A; B, C | G)", 1)
        end
    end
    return nothing
end

function rule_info_sum(s::Sum, ::SimplifyContext)
    parts = map(_coef, s.terms)
    for i in eachindex(parts), j in eachindex(parts)
        i == j && continue
        (ci, ri), (cj, rj) = parts[i], parts[j]
        (ri isa Union{_InfoTerm}) && (rj isa Union{_InfoTerm}) || continue
        same = ci == cj
        (same || ci == -cj) || continue
        out = _info_pair(ri, rj, same)
        out === nothing && continue
        new = mul(Literal(ci), out[1])
        others = PExpr[s.terms[k] for k in eachindex(s.terms) if k != i && k != j]
        return (add(new, others...), out[2])
    end
    return nothing
end
rule_info_sum(::PExpr, ::SimplifyContext) = nothing

push!(_RULES, rule_entropy, rule_mutualinfo, rule_divergence, rule_log, rule_info_sum)

# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

# The distribution of the variables `T` (event variables over true/false).
_varsdist(nets, p::Prob) = _cond_factor(_net(nets, p.name), p.name, _rvsof(p.targets), Atom[], p.given)

_xlog(a, b) = a == 0 ? 0.0 : a * log(b)

function _entropy_factor(nets::_Nets, name::Symbol, T::Vector{RV}, given::Vector{Atom})
    bn = _net(nets, name)
    avg = RV[v for v in (freerv(a) for a in given if !(a isa Intervention)) if v !== nothing]
    fixed = _fixed(given)
    J = _cond_factor(bn, name, vcat(T, avg), Atom[], fixed)       # P(T, avg | fixed)
    C = _cond_factor(bn, name, T, Atom[], given)                  # P(T | avg, fixed)
    return sumout(_combine((j, c) -> -_xlog(j, c), J, C), vcat(T, avg))
end

_eval(h::Entropy, nets::_Nets) = _entropy_factor(nets, h.name, _rvsof(h.X), h.given)
function _eval(m::MutualInformation, nets::_Nets)
    X, Y = _rvsof(m.X), _rvsof(m.Y)
    return _entropy_factor(nets, m.name, X, m.given) + _entropy_factor(nets, m.name, Y, m.given) -
           _entropy_factor(nets, m.name, unique(vcat(X, Y)), m.given)
end
function _eval(d::Divergence, nets::_Nets)
    Pf, Qf = _varsdist(nets, d.p), _varsdist(nets, d.q)
    T = _rvsof(d.p.targets)
    if d.kind === :kl
        return sumout(_combine((a, b) -> _xlog(a, a) - _xlog(a, b), Pf, Qf), T)
    elseif d.kind === :cross
        return sumout(_combine((a, b) -> -_xlog(a, b), Pf, Qf), T)
    elseif d.kind === :chisq
        return sumout(_combine((a, b) -> a == 0 ? 0.0 : a^2 / b, Pf, Qf), T) - _scalar_factor(1.0)
    else
        α = d.α
        S = sumout(_combine((a, b) -> a == 0 ? 0.0 : a^α * b^(1 - α), Pf, Qf), T)
        return _fmap(x -> log(x) / (α - 1), S)
    end
end
_eval(l::Log, nets::_Nets) = _fmap(log, _eval(l.arg, nets))

# ---------------------------------------------------------------------------
# Sequences: transfer entropy and directed information (definitions)
# ---------------------------------------------------------------------------

"""
    transfer_entropy(x, y, t; lag = 1, name = :P) -> MutualInformation

The transfer entropy from the sequence `x` to `y` at time `t` (an
[`Index`](@ref)) with history length `lag`:
`T_{x→y} = I(y[t]; x[t-lag:t-1] | y[t-lag:t-1])`, written with one element per
lag. This is a definition for rewriting; estimating it from data is out of
scope (see Associations.jl).
"""
function transfer_entropy(x::RV, y::RV, t::Index; lag::Integer=1, name::Symbol=:P)
    lag >= 1 || throw(ArgumentError("lag must be at least 1"))
    xs = Atom[x[t-k] for k in lag:-1:1]
    ys = Atom[y[t-k] for k in lag:-1:1]
    return MutualInformation(Atom[y[t]], xs, ys, name)
end

"""
    directed_information(x, y, n::Integer; name = :P) -> PExpr

Massey's directed information `I(x[1:n] → y[1:n]) = ∑ₜ I(x_1, …, x_t; y_t | y_1, …, y_{t-1})`
for a concrete length `n` (elements `x_1, …` are ordinary variables, so the
result can be computed on a network over them).
"""
function directed_information(x::RV, y::RV, n::Integer; name::Symbol=:P)
    return add((MutualInformation(Atom[x[k] for k in 1:t], Atom[y[t]], Atom[y[k] for k in 1:t-1], name)
                for t in 1:n)...)
end

# ---------------------------------------------------------------------------
# Closed forms for distribution families
# ---------------------------------------------------------------------------

# Special functions for random quantities: log Γ and the digamma function ψ,
# written out here so the package needs no SpecialFunctions dependency.
const _LANCZOS = (0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
                  -176.61502916214059, 12.507343278686905, -0.13857109526572012,
                  9.9843695780195716e-6, 1.5056327351493116e-7)
function _loggamma(x::Real)
    x = float(x)
    x < 0.5 && return log(π / abs(sin(π * x))) - _loggamma(1 - x)    # reflection
    x -= 1
    a = _LANCZOS[1]
    t = x + 7.5
    for i in 1:8
        a += _LANCZOS[i+1] / (x + i)
    end
    return 0.5 * log(2π) + (x + 0.5) * log(t) - t + log(a)
end
function _digamma(x::Real)
    x = float(x)
    x <= 0 && isinteger(x) && return NaN
    x < 0 && return _digamma(1 - x) - π / tan(π * x)                # reflection
    r = 0.0
    while x < 6
        r -= 1 / x
        x += 1
    end
    f = 1 / (x * x)
    return r + log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f * (1 / 132)))))
end
# numeric functions behind RFun names (used by compute and evaluate)
_rfunction(f::Symbol) = f === :loggamma ? _loggamma : f === :digamma ? _digamma :
                        f === :Phi ? _normcdf : getfield(Base, f)

const _LXFNAMES = Dict(:loggamma => "\\log\\Gamma", :digamma => "\\psi", :Phi => "\\Phi",
                       :atan => "\\arctan", :sign => "\\operatorname{sign}")
_lxfname(f::Symbol) = get(_LXFNAMES, f, string("\\", f))

# The standard normal CDF Φ, through erfc: a Taylor series for small
# arguments and a continued fraction in the tails (both to ~1e-15).
function _erfc(x::Float64)
    x < 0 && return 2 - _erfc(-x)
    if x < 2.5
        s, t, n = 0.0, x, 0
        while abs(t) > 1e-17 * max(abs(s), 1e-300) || n < 3
            s += t / (2n + 1)
            n += 1
            t *= -x^2 / n
        end
        return 1 - 2 / sqrt(π) * s
    end
    f = 0.0                                       # Lentz-free backward evaluation
    for k in 60:-1:1
        f = (k / 2) / (x + f)
    end
    return exp(-x^2) / sqrt(π) / (x + f)
end
_normcdf(x::Real) = 0.5 * _erfc(-float(x) / sqrt(2))

"""
    Family(name, params...)

A parametric distribution family with symbolic or numeric parameters, for the
closed-form [`H`](@ref) (differential entropy for continuous families) and
[`KL`](@ref):

| family | parameters |
|:--|:--|
| `:Normal` | mean `μ`, standard deviation `σ` |
| `:Laplace` | location `μ`, scale `b` |
| `:Uniform` | `a`, `b` (KL assumes the first support lies inside the second) |
| `:Exponential` | rate `λ` |
| `:Gamma` | shape `α`, rate `β` |
| `:Beta` | `α`, `β` |
| `:Dirichlet` | a vector of concentrations `[α₁, …, αₖ]` |
| `:Poisson` | rate `λ` (KL only) |
| `:Geometric` | success probability `p` (on 1, 2, …) |
| `:NegativeBinomial` | number of successes `r`, success probability `p` (trials until the `r`-th success, on `r`, `r + 1`, …) |
| `:Bernoulli` | `p` |
| `:Binomial` | number of trials `n`, success probability `p` (KL only) |
| `:LogNormal` | `μ`, `σ` of the logarithm |
| `:Cauchy` | location `x₀`, scale `γ` |
| `:MvNormal` | mean vector and covariance matrix, numbers only |

Distributions of transformed variables ([`law`](@ref)), moments, generating
functions ([`MGF`](@ref), [`PGF`](@ref), [`CF`](@ref)), densities
([`PDF`](@ref), [`CDF`](@ref)) use the same families.

Parameters are numbers, variables (`@rv μ::continuous`) or random quantities
(`2μ + 1`); the results are random quantities that fold to numbers when every
parameter is a number, and [`evaluate`](@ref) gives their value at numbers.
`log Γ` and the digamma function `ψ` appear as `loggamma` and `digamma`.
"""
struct Family
    name::Symbol
    params::Vector{Any}
end
const _FAMILIES = Dict(:Normal => 2, :Laplace => 2, :Uniform => 2, :Exponential => 1, :Gamma => 2,
                       :Beta => 2, :Dirichlet => 1, :Poisson => 1, :Geometric => 1, :Bernoulli => 1,
                       :MvNormal => 2, :LogNormal => 2, :Cauchy => 2, :Binomial => 2,
                       :NegativeBinomial => 2)
_fparam(p::Integer) = RConst(float(p))       # floats, so negative powers of integers are fine
_fparam(p) = _r(p)
function Family(name::Symbol, params...)
    haskey(_FAMILIES, name) || throw(ArgumentError(
        "closed forms are implemented for $(join(sort!(string.(collect(keys(_FAMILIES)))), ", ")); got $name"))
    length(params) == _FAMILIES[name] ||
        throw(ArgumentError("$name takes $(_FAMILIES[name]) parameter(s)"))
    if name === :MvNormal
        μ, Σ = params
        (μ isa AbstractVector{<:Real} && Σ isa AbstractMatrix{<:Real} && size(Σ) == (length(μ), length(μ))) ||
            throw(ArgumentError("MvNormal takes a numeric mean vector and a matching covariance matrix"))
        return Family(name, Any[float.(μ), float.(Σ)])
    elseif name in (:Binomial, :NegativeBinomial)   # the count parameter stays an integer

        return Family(name, Any[_r(params[1]), _fparam(params[2])])
    elseif name === :Dirichlet
        αs = params[1]

        (αs isa AbstractVector && length(αs) >= 2) ||
            throw(ArgumentError("Dirichlet takes a vector of at least two concentrations"))
        return Family(name, Any[RExpr[_fparam(a) for a in αs]])
    end
    return Family(name, Any[_fparam(p) for p in params])
end
Base.:(==)(a::Family, b::Family) = a.name === b.name && a.params == b.params
_fstr(p::RExpr) = _rstr(p)
_fstr(p::AbstractVector{<:RExpr}) = string("[", join(map(_rstr, p), ", "), "]")
_fstr(p) = string(p)
Base.show(io::IO, f::Family) = print(io, f.name, "(", join(map(_fstr, f.params), ", "), ")")

# arithmetic on random quantities that folds numbers
function _rf(f::Symbol, x::RExpr)
    x isa RConst && f === :exp && iszero(x.value) && return RConst(1)    # exactly
    x isa RConst && f === :log && isone(x.value) && return RConst(0)
    x isa RConst && return RConst(_rfunction(f)(x.value))

    f === :log && x isa RFun && x.f === :exp && return x.arg       # log(exp(a)) = a
    f === :exp && x isa RFun && x.f === :log && return x.arg       # exp(log(a)) = a
    return RFun(f, x)
end
_rlog(x::RExpr) = _rf(:log, x)
_rlgamma(x::RExpr) = _rf(:loggamma, x)
_rdigamma(x::RExpr) = _rf(:digamma, x)
_rc(x) = RConst(x)
_neg(x) = rmul(_rc(-1), x)
_rsub(a, b) = radd(a, _neg(b))
_rdiv(a, b) = rmul(a, rpow(b, -1))
_rsum(xs) = radd(xs...)
_lbeta(as) = _rsub(_rsum(RExpr[_rlgamma(a) for a in as]), _rlgamma(_rsum(as)))

function H(f::Family)
    n = f.name
    p = f.params
    if n === :Normal
        return radd(_rc(0.5 * log(2π * ℯ)), _rlog(p[2]))
    elseif n === :Laplace
        return radd(_rc(1 + log(2)), _rlog(p[2]))
    elseif n === :Uniform
        return _rlog(_rsub(p[2], p[1]))
    elseif n === :Exponential
        return _rsub(_rc(1), _rlog(p[1]))
    elseif n === :Gamma
        α, β = p
        return radd(α, _neg(_rlog(β)), _rlgamma(α), rmul(_rsub(_rc(1), α), _rdigamma(α)))
    elseif n === :Beta
        a, b = p
        return radd(_lbeta([a, b]), _neg(rmul(_rsub(a, _rc(1)), _rdigamma(a))),
                    _neg(rmul(_rsub(b, _rc(1)), _rdigamma(b))),
                    rmul(_rsub(radd(a, b), _rc(2)), _rdigamma(radd(a, b))))
    elseif n === :Dirichlet
        αs = p[1]
        α0 = _rsum(αs)
        K = length(αs)
        return radd(_lbeta(αs), rmul(_rsub(α0, _rc(K)), _rdigamma(α0)),
                    _neg(_rsum(RExpr[rmul(_rsub(a, _rc(1)), _rdigamma(a)) for a in αs])))
    elseif n === :Geometric
        q = p[1]
        qc = _rsub(_rc(1), q)
        return _neg(_rdiv(radd(rmul(qc, _rlog(qc)), rmul(q, _rlog(q))), q))
    elseif n === :Bernoulli
        q = p[1]
        qc = _rsub(_rc(1), q)
        return _neg(radd(rmul(q, _rlog(q)), rmul(qc, _rlog(qc))))
    elseif n === :MvNormal
        Σ = p[2]
        k = size(Σ, 1)
        return _rc(0.5 * (k * log(2π * ℯ) + logdet(Σ)))
    elseif n === :LogNormal
        return radd(p[1], _rc(0.5 * log(2π * ℯ)), _rlog(p[2]))
    elseif n === :Cauchy
        return radd(_rc(log(4π)), _rlog(p[2]))
    end
    throw(ArgumentError("no closed-form entropy for $(n)"))
end

function KL(f::Family, g::Family)
    f.name === g.name || throw(ArgumentError("closed-form KL needs two members of the same family"))
    n = f.name
    if n === :Normal
        μ1, σ1 = f.params
        μ2, σ2 = g.params
        d = _rsub(μ1, μ2)
        return radd(_rlog(σ2), _neg(_rlog(σ1)),
                    _rdiv(radd(rpow(σ1, 2), rpow(d, 2)), rmul(_rc(2), rpow(σ2, 2))), _rc(-0.5))
    elseif n === :Laplace
        μ1, b1 = f.params
        μ2, b2 = g.params
        d = _rf(:abs, _rsub(μ1, μ2))
        return radd(_rlog(b2), _neg(_rlog(b1)), _rdiv(d, b2),
                    rmul(_rdiv(b1, b2), _rf(:exp, _neg(_rdiv(d, b1)))), _rc(-1))
    elseif n === :Uniform
        a1, b1 = f.params
        a2, b2 = g.params
        return _rsub(_rlog(_rsub(b2, a2)), _rlog(_rsub(b1, a1)))
    elseif n === :Exponential
        λ1, λ2 = f.params[1], g.params[1]
        return radd(_rlog(λ1), _neg(_rlog(λ2)), _rdiv(λ2, λ1), _rc(-1))
    elseif n === :Gamma
        α1, β1 = f.params
        α2, β2 = g.params
        return radd(rmul(_rsub(α1, α2), _rdigamma(α1)), _neg(_rlgamma(α1)), _rlgamma(α2),
                    rmul(α2, _rsub(_rlog(β1), _rlog(β2))), rmul(α1, _rdiv(_rsub(β2, β1), β1)))
    elseif n === :Beta
        a1, b1 = f.params
        a2, b2 = g.params
        return radd(_lbeta([a2, b2]), _neg(_lbeta([a1, b1])), rmul(_rsub(a1, a2), _rdigamma(a1)),
                    rmul(_rsub(b1, b2), _rdigamma(b1)),
                    rmul(_rsub(radd(a2, b2), radd(a1, b1)), _rdigamma(radd(a1, b1))))
    elseif n === :Dirichlet
        α, β = f.params[1], g.params[1]
        length(α) == length(β) || throw(ArgumentError("the two Dirichlets need the same dimension"))
        α0, β0 = _rsum(α), _rsum(β)
        return radd(_rlgamma(α0), _neg(_rsum(RExpr[_rlgamma(a) for a in α])), _neg(_rlgamma(β0)),
                    _rsum(RExpr[_rlgamma(b) for b in β]),
                    _rsum(RExpr[rmul(_rsub(a, b), _rsub(_rdigamma(a), _rdigamma(α0))) for (a, b) in zip(α, β)]))
    elseif n === :Poisson
        λ1, λ2 = f.params[1], g.params[1]
        return radd(rmul(λ1, _rsub(_rlog(λ1), _rlog(λ2))), λ2, _neg(λ1))
    elseif n === :Geometric
        p, q = f.params[1], g.params[1]
        pc, qc = _rsub(_rc(1), p), _rsub(_rc(1), q)
        # (1 - p) / p failures before the success, on average
        return radd(_rsub(_rlog(p), _rlog(q)), rmul(_rdiv(pc, p), _rsub(_rlog(pc), _rlog(qc))))
    elseif n === :Bernoulli
        p, q = f.params[1], g.params[1]
        pc, qc = _rsub(_rc(1), p), _rsub(_rc(1), q)
        return radd(rmul(p, _rsub(_rlog(p), _rlog(q))), rmul(pc, _rsub(_rlog(pc), _rlog(qc))))
    elseif n === :MvNormal
        μ1, Σ1 = f.params
        μ2, Σ2 = g.params
        length(μ1) == length(μ2) || throw(ArgumentError("the two MvNormals need the same dimension"))
        k = length(μ1)
        S2 = inv(Σ2)
        d = μ2 - μ1
        return _rc(0.5 * (tr(S2 * Σ1) + dot(d, S2 * d) - k + logdet(Σ2) - logdet(Σ1)))
    elseif n === :LogNormal          # the same as for the logarithms, which are Normal
        return KL(Family(:Normal, f.params...), Family(:Normal, g.params...))
    elseif n === :Cauchy
        x1, γ1 = f.params
        x2, γ2 = g.params
        return _rlog(_rdiv(radd(rpow(radd(γ1, γ2), 2), rpow(_rsub(x1, x2), 2)), rmul(_rc(4), γ1, γ2)))
    elseif n === :Binomial
        f.params[1] == g.params[1] || throw(ArgumentError("closed-form KL needs two Binomials with the same n"))
        return rmul(f.params[1], KL(Family(:Bernoulli, f.params[2]), Family(:Bernoulli, g.params[2])))
    end
    throw(ArgumentError("no closed-form KL for $(n)"))
end


# Evaluate a random quantity at values of its variables (numbers only).
_reval(x::RConst, vals) = x.value
_reval(x::RVal, vals) = vals[x.rv]
_reval(x::RAdd, vals) = sum(_reval(t, vals) for t in x.terms)
_reval(x::RMul, vals) = prod(_reval(t, vals) for t in x.factors)
_reval(x::RPow, vals) = _numpow(_reval(x.base, vals), x.p)

_reval(x::RFun, vals) = _rfunction(x.f)(_reval(x.arg, vals))
"""
    evaluate(r::RExpr, values::Pair...) -> Number

The value of a random quantity (such as a closed-form KL) at given values of
its variables: `evaluate(KL(Family(:Normal, μ, 1), Family(:Normal, 0, 1)), μ => 2)`.
"""
evaluate(r::RExpr, vals::Pair...) = _reval(r, Dict{RV,Any}(vals...))
evaluate(r::RV, vals::Pair...) = evaluate(_r(r), vals...)

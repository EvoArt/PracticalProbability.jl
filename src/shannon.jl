# Proving information inequalities.
#
# A linear inequality in entropies and mutual informations is Shannon-type if
# it follows from the non-negativity of (conditional) entropy and mutual
# information. Written over the 2ⁿ - 1 joint entropies h(S) of the subsets of
# n variables, it holds for every distribution exactly when its coefficient
# vector is a non-negative combination of the elemental inequalities
#
#     H(Xᵢ | X_{N∖i}) ≥ 0,      I(Xᵢ; Xⱼ | X_K) ≥ 0  (K ⊆ N ∖ {i, j})
#
# (Yeung's framework, as in ITIP). Equalities assumed on top (independence
# statements, a graph's d-separations) enter with coefficients of either sign.
# The search for the combination is a non-negative least-squares problem; the
# coefficients found are the proof. When there is none, a search over small
# distributions looks for a counterexample (an inequality can also be true but
# not Shannon-type, such as Zhang–Yeung's).

"""
    ShannonProof

The result of [`prove`](@ref): whether the inequality follows from Shannon's
inequalities (`holds`), and if so the proof, as non-negative weights on
elemental inequalities (`terms`) plus multiples of the assumed equalities
(`assumed`), so that `rhs - lhs = ∑ terms + ∑ assumed`. If not, possibly a
`counterexample`: a joint distribution (a [`Factor`](@ref)) that violates it.
"""
struct ShannonProof
    inequality::Inequality
    holds::Bool
    vars::Vector{RV}
    terms::Vector{Pair{Real,PExpr}}
    assumed::Vector{Pair{Real,PExpr}}
    counterexample::Union{Nothing,Factor}
    violation::Float64
    nonshannon::Vector{Pair{Real,PExpr}}            # Zhang–Yeung instances used
    copies::Vector{Tuple{Vector{RV},Vector{RV}}}    # copy-lemma copies (R, over O) used
end
ShannonProof(q, h, vars, terms, assumed, ce, viol) =
    ShannonProof(q, h, vars, terms, assumed, ce, viol, Pair{Real,PExpr}[], Tuple{Vector{RV},Vector{RV}}[])

holds(p::ShannonProof) = p.holds
Base.Bool(p::ShannonProof) = p.holds

# ---------------------------------------------------------------------------
# Linear forms over joint entropies
# ---------------------------------------------------------------------------

_infovars(e::PExpr) = sort!(collect(_infovarset(e)))
_infovarset(e::PExpr) = mapreduce(_infovarset, union, children(e); init=Set{RV}())
_infovarset(h::Entropy) = Set(_rvsof(vcat(h.X, h.given)))
_infovarset(m::MutualInformation) = Set(_rvsof(vcat(m.X, m.Y, m.given)))
_infovarset(::Literal) = Set{RV}()

_mask(vars, S) = reduce(|, (UInt(1) << (findfirst(==(v), vars) - 1) for v in S); init=UInt(0))

# The expression as (coefficients of h(S) indexed by mask, constant).
function _linear_form(e::PExpr, vars::Vector{RV})
    b = zeros(2^length(vars) - 1)
    c = Ref(0.0)
    names = Set{Symbol}()
    _lin!(b, c, names, e, 1.0, vars)
    length(names) <= 1 || throw(ArgumentError("all terms must be about one distribution"))
    return b, c[]
end

_lin_err(e) = throw(ArgumentError("prove works with linear combinations of entropies and mutual " *
                                  "informations that condition on variables only; cannot read $(e)"))
_lin!(b, c, names, e::PExpr, k, vars) = _lin_err(e)
_lin!(b, c, names, l::Literal, k, vars) = (c[] += k * l.value; nothing)
_lin!(b, c, names, s::Sum, k, vars) = foreach(t -> _lin!(b, c, names, t, k, vars), s.terms)
function _lin!(b, c, names, p::Product, k, vars)
    lits = [f for f in p.factors if f isa Literal]
    rest = [f for f in p.factors if !(f isa Literal)]
    length(rest) == 1 || _lin_err(p)
    _lin!(b, c, names, rest[1], k * prod(f.value for f in lits; init=1), vars)
end
function _hterm!(b, A, C, k, vars)
    _hvec!(b, _mask(vars, union(A, C)), k)
    _hvec!(b, _mask(vars, C), -k)
end
function _lin!(b, c, names, h::Entropy, k, vars)
    all(_isvarlike, h.given) || _lin_err(h)
    push!(names, h.name)
    _hterm!(b, _rvsof(h.X), _rvsof(h.given), k, vars)
end
function _lin!(b, c, names, m::MutualInformation, k, vars)
    all(_isvarlike, m.given) || _lin_err(m)
    push!(names, m.name)
    X, Y, G = _rvsof(m.X), _rvsof(m.Y), _rvsof(m.given)
    _hterm!(b, X, G, k, vars)
    _hterm!(b, Y, G, k, vars)
    _hterm!(b, union(X, Y), G, -k, vars)
end

# h(S) coefficients of H(A | C) and I(A; B | C), with h(∅) = 0
function _hvec!(v, s, k)
    s == 0 || (v[s] += k)
    return v
end
function _mivec(m, a::UInt, b::UInt, c::UInt)
    v = zeros(m)
    _hvec!(v, a | c, 1); _hvec!(v, b | c, 1); _hvec!(v, a | b | c, -1); _hvec!(v, c, -1)
    return v
end

_varsof(vars, mask) = RV[v for (i, v) in enumerate(vars) if (mask >> (i - 1)) & 1 == 1]

# Elemental inequalities: their vectors and the expressions they stand for.
function _elemental(vars::Vector{RV}, name::Symbol)
    n = length(vars)
    m = 2^n - 1
    full = UInt(m)
    cols = Vector{Float64}[]
    exprs = PExpr[]
    for i in 1:n
        bit = UInt(1) << (i - 1)
        v = zeros(m)
        _hvec!(v, full, 1); _hvec!(v, full & ~bit, -1)
        push!(cols, v)
        push!(exprs, Entropy(Atom[vars[i]], Atom[_varsof(vars, full & ~bit)...], name))
    end
    for i in 1:n, j in i+1:n
        bi, bj = UInt(1) << (i - 1), UInt(1) << (j - 1)
        rest = full & ~bi & ~bj
        # every subset K of the rest
        K = rest
        while true
            push!(cols, _mivec(m, bi, bj, K))
            push!(exprs, MutualInformation(Atom[vars[i]], Atom[vars[j]], Atom[_varsof(vars, K)...], name))
            K == 0 && break
            K = (K - 1) & rest
        end
    end
    return cols, exprs
end

# Assumed equalities as (vector, expression) pairs.
_constraint(s::Independence, vars, name) = information(s; name)
_constraint(e::PExpr, vars, name) = e
function _constraint(e::Equation, vars, name)
    e.relation === :eq || throw(ArgumentError("only equalities can be assumed"))
    return add(e.lhs, -e.rhs)
end
_constraint(x, vars, name) = throw(ArgumentError(
    "assumptions are independence statements, expressions that are zero, or equations; got $(repr(x))"))

# Every conditional independence among `vars` that the graph implies.
function _graph_constraints(g, vars::Vector{RV})
    out = Independence[]
    n = length(vars)
    for i in 1:n, j in i+1:n
        others = RV[v for v in vars if v != vars[i] && v != vars[j]]
        for k in 0:length(others), K in _combinations(others, k)
            d_separated(g, vars[i], vars[j], K) && push!(out, Independence(RV[vars[i]], RV[vars[j]], K))
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# Non-negative least squares (Lawson–Hanson)
# ---------------------------------------------------------------------------

function _nnls(A::AbstractMatrix, b::AbstractVector; tol=1e-10)
    m, n = size(A)
    x = zeros(n)
    passive = falses(n)
    w = A' * (b - A * x)
    for _ in 1:3n
        cand = [!passive[j] && w[j] > tol for j in 1:n]
        any(cand) || break
        j = argmax([cand[k] ? w[k] : -Inf for k in 1:n])
        passive[j] = true
        for _ in 1:3n
            idx = findall(passive)
            z = zeros(n)
            z[idx] = A[:, idx] \ b
            if all(z[i] > tol for i in idx)
                x = z
                break
            end
            α = minimum(x[i] / (x[i] - z[i]) for i in idx if z[i] <= tol; init=1.0)
            x = x .+ α .* (z .- x)
            for i in idx
                if x[i] <= tol
                    passive[i] = false
                    x[i] = 0.0
                end
            end
            isempty(findall(passive)) && break
        end
        w = A' * (b - A * x)
    end
    return x
end

# ---------------------------------------------------------------------------
# Counterexamples
# ---------------------------------------------------------------------------

function _joint_entropies(p::Array{Float64}, n::Int)
    m = 2^n - 1
    h = zeros(m)
    for s in 1:m
        keep = [((s >> (i - 1)) & 1) == 1 for i in 1:n]
        dims = Tuple(i for i in 1:n if !keep[i])
        q = isempty(dims) ? p : sum(p; dims=dims)
        h[s] = -sum(x -> x > 0 ? x * log(x) : 0.0, q)
    end
    return h
end

function _counterexample(b, c, vars, rng; trials=3000)
    n = length(vars)
    best = (Inf, nothing)
    for t in 1:trials
        card = t % 3 == 0 ? 3 : 2
        sz = ntuple(_ -> card, n)
        p = if t % 2 == 0
            # a uniform distribution on a random support: functions, XORs, copies
            k = rand(rng, 1:min(8, card^n))
            q = zeros(sz)
            for i in Random.randperm(rng, card^n)[1:k]
                q[i] = 1.0
            end
            q
        else
            rand(rng, sz...) .^ rand(rng, (1, 3, 8))
        end
        p ./= sum(p)
        val = sum(b .* _joint_entropies(p, n)) + c
        val < best[1] && (best = (val, p))
    end
    best[1] < -1e-9 || return (nothing, 0.0)
    p = best[2]
    doms = Vector{Any}[collect(Any, 1:size(p, i)) for i in 1:n]
    return (Factor(copy(vars), doms, p, "counterexample"), -best[1])
end

# ---------------------------------------------------------------------------
# prove
# ---------------------------------------------------------------------------

_nicenum(x::Real) = (r = rationalize(x; tol=1e-9); denominator(r) == 1 ? numerator(r) : r)

"""
    prove(lhs ≤ rhs; assume = [], dag = nothing, rng) -> ShannonProof
    prove(rhs ≥ lhs; …)

Decide whether an information inequality (a linear combination of entropies
`H(⋯ | ⋯)` and mutual informations `MI(⋯)` over plain variables) follows from
the basic (Shannon) inequalities, the ITIP way: rewrite it in joint entropies
and write `rhs - lhs` as a non-negative combination of elemental inequalities
`H(Xᵢ | rest) ≥ 0` and `I(Xᵢ; Xⱼ | K) ≥ 0`. The combination found is the proof.

`assume` adds equalities: independence statements (`X ⟂ Z | Y` means
`I(X; Z | Y) = 0`), expressions that are zero (`H(Y | X)`: `Y` is a function
of `X`) or equations. `dag = g` assumes every conditional independence among
the variables that `g` implies.

When there is no proof, `prove` searches small distributions for a
counterexample. Without one the inequality may still be true but not
Shannon-type (e.g. the Zhang–Yeung inequality). Two ways beyond Shannon:

- `nonshannon = true` adds every instance of the Zhang–Yeung inequality
  `2I(C; D) ≤ I(A; B) + I(A; C, D) + 3I(C; D | A) + I(C; D | B)` (over four of
  the variables) as a further generator;
- `copies = [(R, O), …]` applies the copy lemma (Zhang–Yeung, Dougherty–
  Freiling–Zeger): new variables `R*` with the same joint distribution with `O`
  as `R`, and independent of everything else given `O`. Inequalities about the
  original variables proved with them hold for every distribution.
  `copies = :auto` tries single copies of one or two variables over one or two
  others until one gives a proof.

```julia
prove(H(X, Y) <= H(X) + H(Y))                          # I(X; Y) ≥ 0
prove(MI(X, Z) <= MI(X, Y); assume = [X ⟂ Z | Y])      # data processing
prove(MI(X, Y | Z) <= MI(X, Y))                        # false: Z = X ⊻ Y
```
"""
function prove(q::Inequality; assume=(), dag=nothing, nonshannon::Bool=false, copies=(),
               rng::AbstractRNG=Random.Xoshiro(1))
    e = add(q.rhs, -q.lhs)                     # to show: e ≥ 0
    cons = PExpr[]
    vars = _infovars(e)
    name = _infoname(e)
    for a in (assume isa Union{Tuple,AbstractVector} ? assume : (assume,))
        a isa Facts && (foreach(s -> push!(cons, information(s; name)), a.independent); continue)
        push!(cons, _constraint(a, vars, name))
    end
    vars = sort!(collect(union(Set(vars), (Set(_infovars(c)) for c in cons)...)))
    if dag !== nothing
        g = _asdag(dag)
        all(v -> v in g.nodes, vars) || throw(ArgumentError("every variable must be a node of the graph"))
        append!(cons, (information(s; name) for s in _graph_constraints(g, vars)))
    end
    isempty(vars) && throw(ArgumentError("the inequality mentions no variables"))
    if copies === :auto
        r = _prove_core(q, e, vars, cons, name, nonshannon, Tuple{Vector{RV},Vector{RV}}[])
        r.holds && return r
        for cp in _copy_candidates(vars)
            length(vars) + length(cp[1]) <= 8 || continue
            r = _prove_core(q, e, vars, cons, name, nonshannon, [cp])
            r.holds && return r
        end
        return _prove_core(q, e, vars, cons, name, nonshannon, Tuple{Vector{RV},Vector{RV}}[]; search=true, rng)
    end
    cps = Tuple{Vector{RV},Vector{RV}}[(_rvlist(R), _rvlist(O)) for (R, O) in copies]
    return _prove_core(q, e, vars, cons, name, nonshannon, cps; search=true, rng)
end

# Copies to try for `copies = :auto`: one or two variables copied over one or
# two others (the shape of the Zhang–Yeung and Dougherty–Freiling–Zeger proofs).
function _copy_candidates(vars)
    out = Tuple{Vector{RV},Vector{RV}}[]
    for kr in 1:min(2, length(vars) - 1), R in _combinations(vars, kr)
        rest = RV[v for v in vars if !(v in R)]
        for ko in 1:min(2, length(rest)), O in _combinations(rest, ko)
            push!(out, (R, O))
        end
    end
    return out
end

_copyvar(v::RV) = RV(Symbol(v.name, "*"), v.kind)

# The copy lemma: for R copied over O there are variables R* with
# (R*, O) distributed as (R, O) and R* ⟂ (everything else) | O.
function _copy_constraints(R::Vector{RV}, O::Vector{RV}, vars::Vector{RV}, name::Symbol)
    cons = PExpr[]
    Rs = RV[_copyvar(v) for v in R]
    for kt in 1:length(R), T in _combinations(collect(eachindex(R)), kt), ku in 0:length(O), U in _combinations(O, ku)
        push!(cons, add(Entropy(Atom[Rs[T]..., U...], Atom[], name), -Entropy(Atom[R[T]..., U...], Atom[], name)))
    end
    others = RV[v for v in vars if !(v in O)]
    isempty(others) || push!(cons, MutualInformation(Atom[Rs...], Atom[others...], Atom[O...], name))
    return Rs, cons
end

# Zhang–Yeung: 2I(C;D) ≤ I(A;B) + I(A;C,D) + 3I(C;D|A) + I(C;D|B), as rhs - lhs ≥ 0.
_zy(A, B, C, D, name) = add(MutualInformation(Atom[A], Atom[B], Atom[], name),
                            MutualInformation(Atom[A], Atom[C, D], Atom[], name),
                            mul(Literal(3), MutualInformation(Atom[C], Atom[D], Atom[A], name)),
                            MutualInformation(Atom[C], Atom[D], Atom[B], name),
                            mul(Literal(-2), MutualInformation(Atom[C], Atom[D], Atom[], name)))

function _prove_core(q, e, vars0, cons0, name, nonshannon, cps; search::Bool=false,
                     rng::AbstractRNG=Random.Xoshiro(1))
    vars = copy(vars0)
    cons = copy(cons0)
    ncons_user = length(cons0)
    for (R, O) in cps
        all(v -> v in vars0, vcat(R, O)) || throw(ArgumentError("copies must be of variables in the inequality"))
        Rs, cc = _copy_constraints(R, O, vars0, name)
        append!(vars, Rs)
        append!(cons, cc)
    end
    length(vars) <= 8 || throw(ArgumentError("prove handles up to 8 variables (copies included)"))
    b, c = _linear_form(e, vars)
    cols, exprs = _elemental(vars, name)
    nelem = length(cols)
    zyexprs = PExpr[]
    if nonshannon && length(vars) >= 4
        for A in vars, B in vars, C in vars, D in vars
            length(unique([A, B, C, D])) == 4 && C < D || continue
            z = _zy(A, B, C, D, name)
            push!(cols, _linear_form(z, vars)[1])
            push!(exprs, z)
            push!(zyexprs, z)
        end
    end
    ngen = length(cols)
    for k in cons
        v, kc = _linear_form(k, vars)
        abs(kc) < 1e-12 || throw(ArgumentError("an assumed equality has a constant term: $(k)"))
        push!(cols, v); push!(exprs, k)
        push!(cols, -v); push!(exprs, k)
    end
    A = reduce(hcat, cols)
    λ = _nnls(A, b)
    resid = maximum(abs, A * λ - b; init=0.0)
    if c >= -1e-12 && resid < 1e-8 * max(1, maximum(abs, b; init=0.0))
        terms = Pair{Real,PExpr}[_nicenum(λ[k]) => exprs[k] for k in 1:nelem if λ[k] > 1e-9]
        zy = Pair{Real,PExpr}[_nicenum(λ[k]) => exprs[k] for k in nelem+1:ngen if λ[k] > 1e-9]
        net = Dict{Int,Float64}()
        for k in ngen+1:length(cols)
            j = ngen + 2 * ((k - ngen - 1) ÷ 2) + 1          # the + column of the pair
            net[j] = get(net, j, 0.0) + (iseven(k - ngen) ? -λ[k] : λ[k])
        end
        assumed = Pair{Real,PExpr}[_nicenum(v) => exprs[j] for (j, v) in sort!(collect(net)) if abs(v) > 1e-9]
        return ShannonProof(q, true, vars, terms, assumed, nothing, 0.0, zy, cps)
    end
    # look for a distribution that breaks it (only without assumptions, which
    # random distributions would not satisfy)
    ce, viol = (search && ncons_user == 0) ? _counterexample(b[1:2^length(vars0)-1], c, vars0, rng) : (nothing, 0.0)
    return ShannonProof(q, false, vars0, Pair{Real,PExpr}[], Pair{Real,PExpr}[], ce, viol)
end

_infoname(e::PExpr) = (ns = _infonames(e); isempty(ns) ? :P : first(ns))
_infonames(e::PExpr) = mapreduce(_infonames, union, children(e); init=Set{Symbol}())
_infonames(h::Union{Entropy,MutualInformation}) = Set([h.name])
_infonames(::Literal) = Set{Symbol}()

"""
    isshannon(ineq; kwargs...) -> Bool

Whether the inequality follows from Shannon's inequalities (see [`prove`](@ref)).
"""
isshannon(q::Inequality; kwargs...) = prove(q; kwargs...).holds

_numstr(x) = x isa Rational ? string(numerator(x), "/", denominator(x)) : string(x)

function Base.show(io::IO, ::MIME"text/plain", p::ShannonProof)
    q = p.inequality
    if p.holds
        how = !isempty(p.copies) ? "Proved with the copy lemma" :
              !isempty(p.nonshannon) ? "Proved with the Zhang–Yeung inequality" : "Shannon-type inequality, proved"
        println(io, how, ": ", q)
        for (R, O) in p.copies
            println(io, "  copy: ", _str(map(_copyvar, R)), " is a copy of ", _str(R), " over ", _str(O),
                    " (same joint distribution with ", _str(O), ", independent of the rest given ", _str(O), ")")
        end
        print(io, "  ", add(q.rhs, -q.lhs), " = ")
        s = ""
        for (k, e) in vcat(p.terms, p.nonshannon, p.assumed)
            a = abs(k)
            body = (a == 1 ? "" : _numstr(a) * " ") * (e isa Sum ? "[" * sprint(show, e) * "]" : sprint(show, e))
            s *= s == "" ? (k < 0 ? "-" : "") * body : (k < 0 ? " - " : " + ") * body
        end
        print(io, s == "" ? "0" : s, " ≥ 0")
        isempty(p.assumed) ||
            print(io, "\n  where ", join((sprint(show, e) * " = 0" for (_, e) in p.assumed), ", "),
                  isempty(p.copies) ? " (assumed)" : " (assumed, or by the copy construction)")
        isempty(p.nonshannon) || print(io, "\n  and each bracketed term is an instance of Zhang–Yeung, ≥ 0")
    elseif p.counterexample !== nothing
        println(io, "False: ", q)
        print(io, "  a distribution of ", _str(p.vars), " on which lhs - rhs = ", round(p.violation; sigdigits=4),
              " nats (see .counterexample)")
    else
        println(io, "Not proved: ", q)
        print(io, "  it does not follow from the basic (Shannon) inequalities", isempty(p.assumed) ? "" : " and the assumptions",
              "; no counterexample found. It may still be true: try `nonshannon = true` or `copies = :auto`")
    end
end
Base.show(io::IO, p::ShannonProof) = print(io, "ShannonProof(", p.inequality, ", holds = ", p.holds, ")")

"""
    proof_expr(p::ShannonProof) -> PExpr

The proof as one expression, `∑ λₖ eₖ` over the elemental inequalities and
assumed equalities used, equal to `rhs - lhs`.
"""
proof_expr(p::ShannonProof) = add((mul(Literal(k), e) for (k, e) in vcat(p.terms, p.nonshannon, p.assumed))...)

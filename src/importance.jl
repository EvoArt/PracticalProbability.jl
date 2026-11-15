# Importance sampling.
#
# An importance weight is the ratio of a target density to a proposal density,
# w = π(z) / q(z). Both are probability expressions here, usually of different
# named distributions (`@density q`), and the weight is derived the way it is
# on paper: write each term in its distribution's factorisation, collect one
# fraction, cancel what target and proposal share, and drop the factors that
# do not depend on the particles (self-normalised weights). Every step keeps
# its justification, and every identity can be checked exactly on discrete
# networks. With generative models, the weight compiles to a log-weight
# function built from the models' own `~` statements.

# ---------------------------------------------------------------------------
# Fractions and constant factors
# ---------------------------------------------------------------------------

# Numerator and denominator factors of an expression, nested quotients flattened.
function _numden(e::PExpr)
    if e isa Quotient
        n1, d1 = _numden(e.num)
        n2, d2 = _numden(e.den)
        return (vcat(n1, d2), vcat(d1, n2))
    elseif e isa Product
        n = PExpr[]
        d = PExpr[]
        for f in e.factors
            a, b = _numden(f)
            append!(n, a)
            append!(d, b)
        end
        return (n, d)
    end
    return (PExpr[e], PExpr[])
end
_fraction(n, d) = divide(mul(n...), mul(d...))

# Split `e` into (factors that depend on `vs`, the constant rest), so that
# e = rest × kept.
function _split_constant(e::PExpr, vs::Set{RV})
    n, d = _numden(e)
    keep(f) = !(f isa Literal) && !isdisjoint(_freevars(f), vs)
    kept = _fraction(filter(keep, n), filter(keep, d))
    rest = _fraction(filter(!keep, n), filter(!keep, d))
    return (kept, rest)
end

"""
    proportional(expr, vars) -> Equation

`expr ∝ kept`, where `kept` is `expr` without the factors that do not depend
on `vars` (as a function of `vars`, the dropped factors are a constant).
Nested fractions are flattened first, so `[P(θ) P(y | θ) / P(y)] / q(θ)`
becomes `P(θ) P(y | θ) / q(θ)` as a function of `θ`.
"""
function proportional(e::PExpr, vars)
    kept, _ = _split_constant(e, Set(_rvlist(vars)))
    return Equation(e, kept; relation=:propto)
end

# ---------------------------------------------------------------------------
# Importance weights
# ---------------------------------------------------------------------------

"""
    ImportanceWeight

The result of [`importance_weight`](@ref):

- `target`, `proposal`: the two distributions;
- `sampled`: the variables drawn from the proposal;
- `data`: variables held fixed (observations); factors in them alone are
  constants;
- `steps`: a [`Derivation`](@ref) of `target / proposal = …`, exact at every
  step (where the proposal is positive);
- `weight`: the weight to compute, `last(steps)` without its constant factors
  (equal to `last(steps)` when `normalized = true`);
- `constant`: what was dropped, `last(steps) = constant × weight`.

Get the weight with [`weight`](@ref). Prints as a derivation ending in `∝`, and
renders as LaTeX. Check it with [`verify`](@ref), [`check_support`](@ref),
[`weight_moments`](@ref) and [`relative_ess`](@ref); compile it with
[`logweight`](@ref) when it came from generative models.
"""
struct ImportanceWeight
    target::PExpr
    proposal::PExpr
    sampled::Vector{RV}
    data::Vector{RV}
    steps::Derivation
    weight::PExpr
    constant::PExpr
    target_model::Union{Nothing,GenerativeModel}
    proposal_model::Union{Nothing,GenerativeModel}
end

"""
    weight(iw::ImportanceWeight) -> PExpr

The importance weight, up to factors that are constant across particles.
"""
weight(iw::ImportanceWeight) = iw.weight

_dagdict(::Nothing) = Dict{Symbol,DAG}()
_dagdict(g::DAG) = Dict{Symbol,DAG}(:P => g)
_dagdict(d::AbstractDict) = Dict{Symbol,DAG}(Symbol(k) => v for (k, v) in d)

function _probs(e::PExpr, out=Prob[])
    e isa Prob ? push!(out, e) : foreach(c -> _probs(c, out), children(e))
    return out
end

_default_data(t::Prob) = _rvsof(Atom[a for a in t.given if !(a isa Intervention)])
_default_data(::PExpr) = RV[]

function _default_sampled(q::PExpr, data)
    vs = RV[v for p in _probs(q) for v in (freerv(a) for a in p.targets) if v !== nothing]
    return unique(RV[v for v in vs if !(v in data)])
end

# A term that is already one of the graph's factors, P(v | pa(v)).
function _isfactor(p::Prob, g::DAG)
    length(p.targets) == 1 && p.targets[1] isa RV || return false
    v = p.targets[1]
    v in g.nodes || return false
    return _atomset(p.given) == _atomset(parents(g, v))
end

function _expand_term(p::Prob, dags)
    g = get(dags, p.name, nothing)
    g === nothing && return p
    _isfactor(p, g) && return p
    all(v -> v in g.nodes, _rvsof(vcat(p.targets, p.given))) || return p
    return expand(p, g)
end

"""
    importance_weight(target, proposal; dag = nothing, data, sampled, normalized = false)
        -> ImportanceWeight

The importance weight `w = target / proposal`, derived symbolically.

`target` and `proposal` are probability expressions. The proposal is usually
written with a named distribution (`@density q`, then `q(θ | y)`), and may
reuse terms of the target's distribution: a proposal `P(θ)` is the target's
own prior, `P(X₂ | X₁)` its own transition kernel. Terms of different
distributions never cancel; identical terms do.

Steps, each recorded with its justification:

1. With no graph for the target, a conditional target `P(θ | y)` is written
   `P(θ, y) / P(y)`.
2. With `dag` (a `DAG` for the distribution `:P`, or a `Dict` from names to
   DAGs), every term is written in its distribution's factorisation
   (see [`expand`](@ref)); terms that already are factors stay as they are.
3. Everything is collected into one fraction, and common factors cancel
   (see [`simplify`](@ref)).
4. Unless `normalized = true`, factors that depend only on `data` (by default
   the conditioning variables of a `Prob` target) are dropped: the weight is
   then known up to a constant, as in self-normalised importance sampling.

`sampled` (by default, the free targets of the proposal's terms that are not
data) only labels the result; everything the weight depends on other than
`data` counts as a particle variable, so in sequential Monte Carlo pass the
observation as `data` and the previous state stays in the weight.

```julia
@rv θ Y
@density q
g = @dag θ → Y
importance_weight(P(θ | Y), P(θ); dag = g)       # w ∝ P(Y | θ)
importance_weight(P(θ | Y), q(θ); dag = g)       # w ∝ P(θ) P(Y | θ) / q(θ)
```

Also accepts two generative models (anything [`generative`](@ref) does): see
the method for models below.
"""
function importance_weight(target::PExpr, proposal::PExpr; dag=nothing,
                           data=_default_data(target), sampled=nothing,
                           normalized::Bool=false)
    dags = _dagdict(dag)
    data = _rvlist(data)
    sampled = sampled === nothing ? _default_sampled(proposal, data) : _rvlist(sampled)
    d = derivation(divide(target, proposal))
    if target isa Prob && !haskey(dags, target.name)
        obs = Atom[a for a in target.given if !(a isa Intervention)]
        if !isempty(obs)
            ints = Atom[a for a in target.given if a isa Intervention]
            jt = _like(target, vcat(target.targets, obs), ints)
            mg = _like(target, obs, ints)
            _push!(d, divide(divide(jt, mg), proposal), "definition of conditional probability")
        end
    end
    if !isempty(dags)
        _push!(d, mapprobs(p -> _expand_term(p, dags), last(d)), "factorisation of each distribution's DAG")
    end
    _push!(d, _fraction(_numden(last(d))...), "collect into one fraction")
    apply!(d, x -> simplify(x; dag=isempty(dags) ? nothing : dags, combine=false, trace=true))
    if normalized
        w, c = last(d), Literal(1)
    else
        particles = Set(RV[v for v in union(freevars(last(d)), sampled) if !(v in data)])
        w, c = _split_constant(last(d), particles)
    end
    return ImportanceWeight(target, proposal, sampled, data, d, w, c, nothing, nothing)
end

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_wname(iw::ImportanceWeight) = isempty(iw.sampled) ? "w" : string("w(", join((v.name for v in iw.sampled), ", "), ")")
_dropped(iw::ImportanceWeight) = iw.constant == Literal(1) ? nothing :
                                 string("drop factors constant across particles: ", sprint(show, iw.constant))

Base.show(io::IO, iw::ImportanceWeight) =
    print(io, _wname(iw), iw.constant == Literal(1) ? " = " : " ∝ ", iw.weight)

function Base.show(io::IO, ::MIME"text/plain", iw::ImportanceWeight)
    println(io, "Importance weight: sample ", join((v.name for v in iw.sampled), ", "),
            " from ", iw.proposal, " for the target ", iw.target)
    lhs = _wname(iw)
    pad = " "^length(lhs)
    rows = Tuple{String,String,String}[]
    push!(rows, (lhs * " = ", sprint(show, first(iw.steps)), ""))
    for i in 2:length(iw.steps)
        push!(rows, (pad * " = ", sprint(show, iw.steps.exprs[i]), iw.steps.rules[i]))
    end
    if iw.constant != Literal(1)
        push!(rows, (pad * " ∝ ", sprint(show, iw.weight), _dropped(iw)))
    end
    width = maximum(r -> textwidth(r[2]), rows)
    for (k, (a, b, c)) in enumerate(rows)
        print(io, "  ", a, b)
        isempty(c) || print(io, " "^(width - textwidth(b) + 3), "[", c, "]")
        k < length(rows) && println(io)
    end
end

function _latex(iw::ImportanceWeight)
    w = isempty(iw.sampled) ? "w" : string("w(", join((_lx(v.name) for v in iw.sampled), ", "), ")")
    lines = String[string(w, " &= ", _latex(first(iw.steps)))]
    for i in 2:length(iw.steps)
        push!(lines, string("&= ", _latex(iw.steps.exprs[i]), " && ", _lxrule(iw.steps.rules[i])))
    end
    if iw.constant != Literal(1)
        push!(lines, string("&\\propto ", _latex(iw.weight), " && ",
                            _lxrule("constant across particles: ") * _latex(iw.constant)))
    end
    return string("\\begin{aligned}\n", join(lines, " \\\\\n"), "\n\\end{aligned}")
end
latex(iw::ImportanceWeight; env::Symbol=:inline) = _wrap(_latex(iw), env)
Base.show(io::IO, ::MIME"text/latex", iw::ImportanceWeight) = print(io, "\$\$", _latex(iw), "\$\$")
Latexify.latexify(iw::ImportanceWeight; env::Symbol=:inline, kwargs...) = latex(iw; env)

Equation(iw::ImportanceWeight) =
    Equation(first(iw.steps), iw.weight; relation=iw.constant == Literal(1) ? :eq : :propto)

# ---------------------------------------------------------------------------
# Numbers: exact checks on discrete networks
# ---------------------------------------------------------------------------

_asfactor(x::Real) = _scalar_factor(x)
_asfactor(f::Factor) = f

# Combine factors entrywise over the union of their variables.
function _broadcast_factors(op, fs::Factor...)
    vars = reduce(union, (f.vars for f in fs); init=RV[])
    doms = Vector{Any}[]
    for v in vars
        f = fs[something(findfirst(f -> v in f.vars, fs))]
        push!(doms, f.domains[something(findfirst(==(v), f.vars))])
    end
    vals = broadcast(op, (_align(f, vars) for f in fs)...)
    vals = vals isa AbstractArray ? Array{Float64}(vals) : fill(Float64(vals))
    if size(vals) != Tuple(length.(doms))         # a variable only in a singleton slot
        vals = vals .* ones(Tuple(length.(doms)))
    end
    return Factor(vars, doms, vals)
end

_unfactor(f::Factor) = isempty(f.vars) ? f.values[] : f

"""
    check_support(iw, nets) -> Bool

Whether the proposal is positive wherever the target is, the condition for
importance sampling to be unbiased. `nets` is a `Dict` from distribution names
to [`BayesNet`](@ref)s, as for [`compute`](@ref).
"""
function check_support(iw::ImportanceWeight, nets::_Nets)
    T = _asfactor(compute(iw.target, nets))
    Q = _asfactor(compute(iw.proposal, nets))
    ok = _broadcast_factors((t, q) -> !(t > 0) || q > 0, T, Q)
    return all(x -> x == 1, ok.values)
end

function _agree_on(mask::Factor, a, b; atol)
    f = _broadcast_factors((m, x, y) -> !(m > 0) || isapprox(x, y; atol) || (isnan(x) && isnan(y)),
                           mask, _asfactor(a), _asfactor(b))
    return all(x -> x == 1, f.values)
end

"""
    verify(iw::ImportanceWeight, nets; atol = 1e-10) -> Bool

Check an importance weight exactly, by enumeration:

1. every step of the derivation equals `target / proposal`;
2. `constant × weight` equals the last step;
3. `proposal × w = target`, the identity behind `E_q[w f] = E_π[f]`;
4. the proposal covers the target's support ([`check_support`](@ref)).

1–3 are checked where the proposal is positive (cancelling `P(θ) / P(θ)` is
valid only there).
"""
function verify(iw::ImportanceWeight, nets::_Nets; atol::Real=1e-10)
    Q = _asfactor(compute(iw.proposal, nets))
    w0 = compute(first(iw.steps), nets)
    all(e -> _agree_on(Q, w0, compute(e, nets); atol), iw.steps.exprs) || return false
    _agree_on(Q, compute(last(iw.steps), nets), compute(mul(iw.constant, iw.weight), nets); atol) || return false
    _agree_on(Q, compute(mul(iw.proposal, last(iw.steps)), nets), compute(iw.target, nets); atol) || return false
    return check_support(iw, nets)
end

"""
    weight_moments(iw, nets) -> (E_q[w], E_q[w²])

The first two moments of the weight under the proposal, summed exactly over
the sampled variables. Numbers when nothing else is free, otherwise
[`Factor`](@ref)s over the remaining variables (data, earlier particles).
"""
function weight_moments(iw::ImportanceWeight, nets::_Nets)
    Q = _asfactor(compute(iw.proposal, nets))
    W = _asfactor(compute(iw.weight, nets))
    m1 = _broadcast_factors((q, w) -> q == 0 ? 0.0 : q * w, Q, W)
    m2 = _broadcast_factors((q, w) -> q == 0 ? 0.0 : q * w^2, Q, W)
    s = RV[v for v in iw.sampled if v in m1.vars]
    return (_unfactor(sumout(m1, s)), _unfactor(sumout(m2, s)))
end

"""
    relative_ess(iw, nets) -> Float64 or Factor

`E_q[w]² / E_q[w²]`, in `(0, 1]`: the effective sample size per particle in
the large-sample limit. `1` means the proposal is the target. Unchanged by the
constant factors dropped from the weight.
"""
function relative_ess(iw::ImportanceWeight, nets::_Nets)
    m1, m2 = weight_moments(iw, nets)
    m1 isa Real && return m1^2 / m2
    return _unfactor(_broadcast_factors((a, b) -> a^2 / b, _asfactor(m1), _asfactor(m2)))
end

# ---------------------------------------------------------------------------
# Generative models: a proposal model, matching `~` lines, compiled log weight
# ---------------------------------------------------------------------------

_tildes(m::GenerativeModel, v::Symbol) = [l for l in m.lines if l.kind === :tilde && l.node === v]
_linekey(l::ModelLine) = (l.lhs, l.rhs, l.dot, l.loops)

# The deterministic assignments a line reads, transitively.
function _detlines(m::GenerativeModel, l::ModelLine, seen=Set{Symbol}())
    out = ModelLine[]
    for s in _readsyms(l.rhs)
        s in seen && continue
        push!(seen, s)
        for a in m.lines
            a.kind === :assign && a.node === s || continue
            append!(out, _detlines(m, a, seen))
            push!(out, a)
        end
    end
    return out
end

# Same distribution for `v` in both models: identical `~` lines and identical
# deterministic definitions behind them.
function _samefactor(gm::GenerativeModel, qm::GenerativeModel, v::Symbol)
    a, b = _tildes(gm, v), _tildes(qm, v)
    length(a) == length(b) || return false
    all(i -> _linekey(a[i]) == _linekey(b[i]), eachindex(a)) || return false
    for i in eachindex(a)
        map(_linekey, _detlines(gm, a[i])) == map(_linekey, _detlines(qm, b[i])) || return false
    end
    return true
end

"""
    importance_weight(target_model, proposal_model; name = :q, normalized = false)
        -> ImportanceWeight

The importance weight for sampling a model's latent variables from another
model. Both are anything [`generative`](@ref) accepts: models defined with
`@graphical` (or their instances, with PracticalBayes), or
[`GenerativeModel`](@ref)s. The target is the model's joint density of its
latent and observed variables, the unnormalised posterior.

The proposal model must `~` every latent variable of the target, and nothing
else. A proposal variable whose `~` statements (and the deterministic
assignments they read) are identical to the target's gets the target's
distribution, so it cancels; any other is a term of the distribution `name`.
Data are the target's observed variables and inputs.

Compile the result with [`logweight`](@ref).
"""
function importance_weight(target, proposal; name::Symbol=:q, normalized::Bool=false)
    gm, qm = generative(target), generative(proposal)
    latent = Set(v.name for v in gm.latent)
    sampled = Set(v.name for v in vcat(qm.latent, qm.observed))
    missing_ = setdiff(latent, sampled)
    isempty(missing_) || throw(ArgumentError("the proposal does not sample $(join(sort!(collect(missing_)), ", "))"))
    extra = setdiff(sampled, latent)
    isempty(extra) || throw(ArgumentError("the proposal samples $(join(sort!(collect(extra)), ", ")), " *
                                          "which are not latent variables of the target"))
    byname = Dict{Symbol,RV}(v.name => v for v in gm.dag.nodes)
    rvof(v::RV) = get(byname, v.name, v)
    factors = PExpr[]
    for v in topological_order(qm.dag)
        v.name in latent || continue
        nm = _samefactor(gm, qm, v.name) ? :P : name
        push!(factors, Prob(Atom[rvof(v)], Atom[rvof(p) for p in parents(qm.dag, v)], nm))
    end
    iw = importance_weight(factorize(gm), mul(factors...);
                           data=vcat(gm.observed, gm.inputs), normalized)
    return ImportanceWeight(iw.target, iw.proposal, iw.sampled, iw.data, iw.steps, iw.weight,
                            iw.constant, gm, qm)
end

const _LW = Symbol("#logw#")

function _logpdf_expr(l::ModelLine, sign::Int)
    term = l.dot ? :(sum(logpdf.($(l.rhs), $(l.lhs)))) : :(logpdf($(l.rhs), $(l.lhs)))
    ex = sign > 0 ? :($_LW += $term) : :($_LW -= $term)
    return _inloops(ex, l)
end
function _inloops(ex, l::ModelLine)
    for (i, r) in reverse(l.loops)
        ex = :(for $i in $r
            $ex
        end)
    end
    return ex
end

# One model's contribution: bind its parameters and data, replay its
# deterministic assignments, add (or subtract) the log densities of `nodes`.
function _model_block(m::GenerativeModel, latent::Set{Symbol}, nodes::Dict{Symbol,Int})
    body = Expr(:block)
    for n in sort!(collect(latent))
        push!(body.args, :($n = getproperty(θ, $(QuoteNode(n)))))
    end
    for a in m.args
        a in latent && continue
        push!(body.args, :($a = $_getdata(data, $(QuoteNode(a)))))
    end
    for l in m.lines
        if l.kind === :assign
            push!(body.args, _inloops(Expr(l.dot ? :.= : :(=), l.lhs, l.rhs), l))
        elseif haskey(nodes, l.node)
            push!(body.args, _logpdf_expr(l, nodes[l.node]))
        end
    end
    return Expr(:let, Expr(:block), body)
end

function _getdata(data, a::Symbol)
    hasproperty(data, a) || throw(ArgumentError("the log weight needs the data `$a`; pass it in the data NamedTuple"))
    return getproperty(data, a)
end

"""
    logweight_expr(iw::ImportanceWeight) -> Expr

The Julia code [`logweight`](@ref) compiles: an anonymous function of
`(θ, data)` that replays each model's deterministic assignments and sums the
`logpdf` of every factor left in the weight (target factors added, proposal
factors subtracted).
"""
function logweight_expr(iw::ImportanceWeight)
    gm, qm = iw.target_model, iw.proposal_model
    (gm === nothing || qm === nothing) &&
        throw(ArgumentError("only weights between two generative models can be compiled"))
    latent = Set(v.name for v in gm.latent)
    tnodes = Dict{Symbol,Int}()
    qnodes = Dict{Symbol,Int}()
    n, d = _numden(iw.weight)
    for (fs, s) in ((n, 1), (d, -1)), f in fs
        f isa Prob && length(f.targets) == 1 && f.targets[1] isa RV ||
            throw(ArgumentError("cannot compile the factor $f"))
        v = f.targets[1].name
        dict = f.name === :P ? tnodes : qnodes
        dict[v] = get(dict, v, 0) + s
    end
    filter!(p -> p.second != 0, tnodes)
    filter!(p -> p.second != 0, qnodes)
    any(p -> abs(p.second) > 1, vcat(collect(tnodes), collect(qnodes))) &&
        throw(ArgumentError("a factor appears more than once in the weight; cannot compile it"))
    return quote
        function (θ, data)
            $_LW = 0.0
            $(_model_block(gm, latent, tnodes))
            $(_model_block(qm, latent, qnodes))
            return $_LW
        end
    end
end

"""
    logweight(iw::ImportanceWeight; mod = Main) -> Function

Compile the log importance weight of a weight between two generative models
into `logw(θ, data)`, where `θ` is a `NamedTuple` of the latent variables and
`data` one of the models' other arguments. Only the factors left after
cancellation are computed: with the prior as proposal, `logw` is the log
likelihood.

The code is evaluated in `mod`, which must have the distributions the models
use and `logpdf` in scope (`using Distributions`). In a notebook (Quarto,
Pluto) or inside a module, code does not run in `Main`: use
[`@logweight`](@ref), which compiles in the calling module. `~` statements
inside `if` blocks are not supported. Inspect the code with
[`logweight_expr`](@ref).
"""
function logweight(iw::ImportanceWeight; mod::Module=Main)
    isdefined(mod, :logpdf) ||
        throw(ArgumentError("logweight evaluates the models' distributions in $(mod), which has no `logpdf`; " *
                            "load Distributions there, pass `mod`, or use `@logweight iw` to compile in the calling module"))
    f = Core.eval(mod, logweight_expr(iw))
    return (θ, data=NamedTuple()) -> Base.invokelatest(f, θ, data)
end

"""
    @logweight iw

[`logweight`](@ref)`(iw; mod = @__MODULE__)`: compile the log weight in the
module the macro is called from, where the models' distributions are in scope.
"""
macro logweight(iw)
    return :($logweight($(esc(iw)); mod=$__module__))
end

# ---------------------------------------------------------------------------
# Sequential Monte Carlo: incremental weights on plates
# ---------------------------------------------------------------------------

"""
    SMCWeight

The result of [`smc_weight`](@ref): the incremental weight of one step of
sequential importance sampling,

    w_t = γ_t(x[1:t]) / [γ_{t-1}(x[1:t-1]) q_t(x[t] | ⋯)],

with `target` the unnormalised path target `γ_t` (a product over a plate),
`previous` its value one step earlier, `proposal` the kernel `q_t`, `steps`
the [`Derivation`](@ref) and `weight` the result. [`path_weight`](@ref) gives
the weight of a whole path, `∏_s w_s`. Check it with [`verify`](@ref) and
[`relative_ess`](@ref) at concrete times.
"""
struct SMCWeight
    target::PlateProduct
    previous::PExpr
    proposal::PExpr
    steps::Derivation
    weight::PExpr
end

weight(sw::SMCWeight) = sw.weight

# IndexedRV atoms ↔ ordinary RVs named after them (x[t-1] ↔ x_t-1), so that
# the graph machinery (which works on RVs) can run on one time slice.
function _freeze(e::PExpr)
    back = Dict{RV,Atom}()
    fa(a) = a isa IndexedRV ? (r = _flatrv(a); back[r] = a; r) : a
    return mapprobs(p -> _like(p, map(fa, p.targets), map(fa, p.given)), e), back
end
_thaw(l::Literal, back) = l
_thaw(p::Prob, back) = _like(p, Atom[get(back, a, a) for a in p.targets], Atom[get(back, a, a) for a in p.given])
_thaw(m::Marginal, back) = marg(m.over, _thaw(m.body, filter(kv -> !(kv.first in m.over), back)))
_thaw(e::PExpr, back) = rebuild(e, PExpr[_thaw(c, back) for c in children(e)])

# The graph of one time slice, read off the factors of f(t): each factor
# P(v | parents) gives a node and its edges. `nothing` if f(t) is not of that form.
function _slice_dag(fs)
    pa = Dict{RV,Vector{RV}}()
    for f in fs
        f isa Prob && f.name === :P && length(f.targets) == 1 && f.targets[1] isa RV &&
            all(a -> a isa RV, f.given) || return nothing
        v = f.targets[1]
        haskey(pa, v) && return nothing
        pa[v] = RV[g for g in f.given]
    end
    isempty(pa) && return nothing
    try
        return DAG(collect(keys(pa)), pa)
    catch
        return nothing                # a cycle: not a factorisation
    end
end

_hasplate(e::PExpr) = e isa PlateProduct || any(_hasplate, children(e))

"""
    smc_weight(target, proposal) -> SMCWeight

The incremental importance weight of sequential Monte Carlo, derived
symbolically. `target` is the unnormalised path target as a product over a
plate, `γ_t = ∏_{s=lo}^{t} f(s)`, typically the joint density of states and
observations up to time `t`; `proposal` is the kernel that extends a particle
by `x[t]`.

The weight `γ_t / (γ_{t-1} q_t)` is reduced by peeling the last factor off
`γ_t` (see [`peel`](@ref)), so `γ_{t-1}` cancels and `f(t) / q_t` remains.
The factors of `f(t)` are then read as the graph of one time slice, and terms
of the target's distribution in the proposal are written in that graph's
factorisation, so the locally optimal proposal `P(x[t] | x[t-1], y[t])`
reduces too.

```julia
@rv x y
@density q
@index t 1:T
@index s 1:t
γ = ∏(s, P(x[s] | x[s-1]) * P(y[s] | x[s]))          # a state-space model
smc_weight(γ, P(x[t] | x[t-1]))            # bootstrap:       P(y[t] | x[t])
smc_weight(γ, P(x[t] | x[t-1], y[t]))      # locally optimal: P(y[t] | x[t-1])
smc_weight(γ, q(x[t] | x[t-1], y[t]))      # anything else
```
"""
function smc_weight(target::PlateProduct, proposal::PExpr)
    pe = peel(target)
    pf = _factors(pe)
    rest, last_ = pf[1] isa PlateProduct ? (pf[1], mul(pf[2:end]...)) : (Literal(1), pe)
    d = derivation(divide(target, mul(rest, proposal)))
    _push!(d, divide(pe, mul(rest, proposal)), "peel the last step off the path target")
    _push!(d, _fraction(_numden(last(d))...), "collect into one fraction")
    apply!(d, x -> simplify(x; combine=false, trace=true))
    # One time slice: the graph of f(t), applied to the target's terms.
    w = last(d)
    if !_hasplate(w)
        fl, _ = _freeze(last_)
        g = _slice_dag(_factors(fl))
        if g !== nothing
            fw, back = _freeze(w)
            dags = Dict{Symbol,DAG}(:P => g)
            ex = mapprobs(p -> _expand_term(p, dags), fw)
            if ex != fw
                _push!(d, _thaw(ex, back), "factorisation of one time slice")
                fr = _fraction(_numden(ex)...)
                _push!(d, _thaw(fr, back), "collect into one fraction")
                s = simplify(fr; dag=dags, combine=false, trace=true)
                for (e, r) in zip(s.exprs[2:end], s.rules[2:end])
                    _push!(d, _thaw(e, back), r)
                end
            end
        end
    end
    return SMCWeight(target, rest, proposal, d, last(d))
end

"""
    path_weight(sw::SMCWeight) -> PExpr

The importance weight of a whole path, the product of the incremental weights
over the target's plate, `∏_{s=lo}^{t} w_s`.
"""
function path_weight(sw::SMCWeight)
    i = sw.target.index
    hi = i.hi
    hi isa Index || throw(ArgumentError("the target's plate must end at an index (such as t) to write the path weight"))
    return ∏(i, _reindex(sw.weight, hi, i))
end

_tname(sw::SMCWeight) = sw.target.index.hi isa Index ? string(sw.target.index.hi.name) : "t"

Base.show(io::IO, sw::SMCWeight) = print(io, "w_", _tname(sw), " = ", sw.weight)

function Base.show(io::IO, ::MIME"text/plain", sw::SMCWeight)
    println(io, "Incremental SMC weight: extend with ", sw.proposal, " for the path target ", sw.target)
    lhs = "w_" * _tname(sw)
    pad = " "^textwidth(lhs)
    rows = Tuple{String,String,String}[(lhs * " = ", sprint(show, first(sw.steps)), "")]
    for k in 2:length(sw.steps)
        push!(rows, (pad * " = ", sprint(show, sw.steps.exprs[k]), sw.steps.rules[k]))
    end
    width = maximum(r -> textwidth(r[2]), rows)
    for (a, b, c) in rows
        print(io, "  ", a, b)
        isempty(c) || print(io, " "^(width - textwidth(b) + 3), "[", c, "]")
        println(io)
    end
    print(io, "  path weight: ")
    pw = try
        sprint(show, path_weight(sw))
    catch
        "the product of the incremental weights"
    end
    print(io, pw)
end

function _latex(sw::SMCWeight)
    w = string("w_{", _lx(Symbol(_tname(sw))), "}")
    lines = String[string(w, " &= ", _latex(first(sw.steps)))]
    for k in 2:length(sw.steps)
        push!(lines, string("&= ", _latex(sw.steps.exprs[k]), " && ", _lxrule(sw.steps.rules[k])))
    end
    return string("\\begin{aligned}\n", join(lines, " \\\\\n"), "\n\\end{aligned}")
end
latex(sw::SMCWeight; env::Symbol=:inline) = _wrap(_latex(sw), env)
Base.show(io::IO, ::MIME"text/latex", sw::SMCWeight) = print(io, "\$\$", _latex(sw), "\$\$")
Latexify.latexify(sw::SMCWeight; env::Symbol=:inline, kwargs...) = latex(sw; env)

# Everything unrolled with the target's end set to the time `k`.
function _at(sw::SMCWeight, k::Integer)
    hi = sw.target.index.hi
    hi isa Union{Index,Symbol} ||
        throw(ArgumentError("the target's plate must end at an index or a symbol to unroll it"))
    u(e) = unroll(e, hi => k)
    return (target=u(sw.target), previous=u(sw.previous), proposal=u(sw.proposal),
            steps=PExpr[u(e) for e in sw.steps.exprs], weight=u(sw.weight))
end

"""
    verify(sw::SMCWeight, nets; at = 2, atol = 1e-10) -> Bool

Check an incremental weight exactly at the time `at` (or at each time in a
collection), on networks over the unrolled variables (`x_0, x_1, …`): every
step agrees, `γ_{t-1} q_t w_t = γ_t`, and `γ_{t-1} q_t` covers `γ_t`. The
equalities are checked where `γ_{t-1} q_t` is positive.
"""
function verify(sw::SMCWeight, nets::_Nets; at=2, atol::Real=1e-10)
    for k in at
        u = _at(sw, k)
        Q = _asfactor(compute(mul(u.previous, u.proposal), nets))
        w0 = compute(u.steps[1], nets)
        all(e -> _agree_on(Q, w0, compute(e, nets); atol), u.steps) || return false
        _agree_on(Q, compute(mul(u.previous, u.proposal, u.weight), nets), compute(u.target, nets); atol) ||
            return false
        T = _asfactor(compute(u.target, nets))
        ok = _broadcast_factors((t, q) -> !(t > 0) || q > 0, T, Q)
        all(x -> x == 1, ok.values) || return false
    end
    return true
end

"""
    relative_ess(sw::SMCWeight, nets; at = 2)

`E_q[w_t]² / E_q[w_t²]` over the new state at the time `at`, for each value of
everything else (earlier states, observations); see
[`relative_ess`](@ref)`(::ImportanceWeight, nets)`.
"""
function relative_ess(sw::SMCWeight, nets::_Nets; at::Integer=2)
    u = _at(sw, at)
    q = u.proposal
    iw = ImportanceWeight(mul(u.weight, q), q, _default_sampled(q, RV[]), RV[],
                          derivation(u.weight), u.weight, Literal(1), nothing, nothing)
    return relative_ess(iw, nets)
end

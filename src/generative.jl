# Generative models: read a Turing-style `@model` body into a graphical model.
#
# A PracticalBayes `Model` (like a Turing one) keeps no trace of its source, so
# the graph is read from the source at definition time: `@graphical` goes in
# front of `@model`, parses the `~` statements, passes the definition through
# untouched, and registers the parsed model against the model function (and
# PracticalBayes' generated evaluator, `_name_eval`). `generative(f_or_model)`
# looks it up.

"""
    ModelLine

One statement of a [`GenerativeModel`](@ref): a `~` statement or a
deterministic assignment, with the loops it sits in.
"""
struct ModelLine
    kind::Symbol            # :tilde | :assign
    lhs::Any
    rhs::Any
    dot::Bool               # `.~` / `.=` (elementwise)
    loops::Vector{Tuple{Symbol,Any}}
    node::Symbol            # base name of the lhs
end

"""
    GenerativeModel

A model's graphical structure, read from its `~` statements by
[`@graphical`](@ref) or [`@generative`](@ref):

- `latent`: `~` variables that are not data (parameters);
- `observed`: `~` variables that are data;
- `inputs`: model arguments used on the right of a `~` that are never `~`'d
  themselves (covariates, sizes, hyperparameters);
- `dag`: the graph over all of these, with deterministic assignments
  (`μ = α + β * x`) resolved through to the variables they are computed from.

Use it with [`posterior`](@ref), [`factorize`](@ref), [`latex`](@ref) and
[`tabulate`](@ref).
"""
struct GenerativeModel
    name::Symbol
    args::Vector{Symbol}
    lines::Vector{ModelLine}
    dag::DAG
    observed::Vector{RV}
    latent::Vector{RV}
    inputs::Vector{RV}
end

const _DISCRETE_DISTS = Set([
    :Bernoulli, :BernoulliLogit, :BernoulliCLogLog, :Binomial, :BinomialLogit, :BetaBinomial,
    :Categorical, :DiscreteUniform, :DiscreteNonParametric, :Geometric, :Hypergeometric,
    :NegativeBinomial, :Poisson, :LogPoisson, :PoissonBinomial, :Skellam, :Multinomial,
    :OrderedLogistic, :Dirac,
])

function _callheads(ex, out=Set{Symbol}())
    if ex isa Expr
        if ex.head === :call && ex.args[1] isa Symbol
            push!(out, ex.args[1])
        elseif ex.head === :. && ex.args[1] isa Symbol && length(ex.args) == 2 && ex.args[2] isa Expr
            push!(out, ex.args[1])
        end
        foreach(a -> _callheads(a, out), ex.args)
    end
    return out
end
_distkind(rhs) = isempty(intersect(_callheads(rhs), _DISCRETE_DISTS)) ? :continuous : :discrete

# Symbols a right-hand side reads (not function names, keyword names or fields).
function _readsyms(ex, out=Set{Symbol}())
    if ex isa Symbol
        push!(out, ex)
    elseif ex isa Expr
        if ex.head === :call
            ex.args[1] isa Symbol || _readsyms(ex.args[1], out)
            foreach(a -> _readsyms(a, out), ex.args[2:end])
        elseif ex.head === :. && length(ex.args) == 2
            if ex.args[2] isa QuoteNode          # x.field
                _readsyms(ex.args[1], out)
            else                                 # f.(args)
                ex.args[1] isa Symbol || _readsyms(ex.args[1], out)
                _readsyms(ex.args[2], out)
            end
        elseif ex.head === :kw || ex.head === :(=)
            _readsyms(ex.args[2], out)
        elseif ex.head === :parameters
            foreach(a -> _readsyms(a, out), ex.args)
        elseif ex.head === :macrocall
            foreach(a -> _readsyms(a, out), ex.args[3:end])
        else
            foreach(a -> _readsyms(a, out), ex.args)
        end
    end
    return out
end

_basename(s::Symbol) = s
_basename(ex::Expr) = ex.head in (:ref, :.) ? _basename(ex.args[1]) :
                      error("@graphical: cannot find the variable in `$ex`")

mutable struct _ParseState
    args::Set{Symbol}
    lines::Vector{ModelLine}
    loops::Vector{Tuple{Symbol,Any}}
    stoch::Vector{Symbol}
    kinds::Dict{Symbol,Symbol}
    parents::Dict{Symbol,Set{Symbol}}
    det::Dict{Symbol,Set{Symbol}}
end

function _resolve(st::_ParseState, syms, self::Union{Nothing,Symbol}=nothing)
    out = Set{Symbol}()
    loopvars = Set(first.(st.loops))
    for s in syms
        s in loopvars && continue
        s === self && continue
        # stochastic first: `theta = Vector(undef, J)` followed by
        # `theta[j] ~ …` makes theta a random variable, not a function of J
        if s in st.stoch
            push!(out, s)
        elseif haskey(st.det, s)
            union!(out, st.det[s])
        elseif s in st.args
            push!(out, s)
        end
    end
    self === nothing || delete!(out, self)
    return out
end

function _loopspecs(spec)
    spec isa Expr && spec.head === :block && return reduce(vcat, map(_loopspecs, spec.args); init=Tuple{Symbol,Any}[])
    if spec isa Expr && spec.head in (:(=), :in) || (spec isa Expr && spec.head === :call && spec.args[1] in (:in, :∈))
        i, r = spec.head === :call ? (spec.args[2], spec.args[3]) : (spec.args[1], spec.args[2])
        return i isa Symbol ? Tuple{Symbol,Any}[(i, r)] : Tuple{Symbol,Any}[]
    end
    return Tuple{Symbol,Any}[]
end

function _walk!(st::_ParseState, ex)
    ex isa Expr || return
    h = ex.head
    if h === :block || h === :local
        foreach(a -> _walk!(st, a), ex.args)
    elseif h === :for
        specs = _loopspecs(ex.args[1])
        append!(st.loops, specs)
        _walk!(st, ex.args[2])
        for _ in specs
            pop!(st.loops)
        end
    elseif h === :while
        _walk!(st, ex.args[2])
    elseif h === :if || h === :elseif
        foreach(a -> _walk!(st, a), ex.args[2:end])
    elseif h === :macrocall
        # `@inbounds for …`, `@views y ~ …`; skip `@addlogprob!` and friends
        name = string(ex.args[1])
        occursin("addlogprob", name) && return
        foreach(a -> _walk!(st, a), ex.args[3:end])
    elseif h === :call && ex.args[1] in (:~, :.~) && length(ex.args) == 3
        _tilde!(st, ex.args[2], ex.args[3], ex.args[1] === :.~)
    elseif (h === :(=) || h === :(:=) || h === :.=) && !(ex.args[1] isa Expr && ex.args[1].head in (:call, :where))
        _assign!(st, ex.args[1], ex.args[2], h === :.=)
    end
    return
end

function _tilde!(st::_ParseState, lhs, rhs, dot::Bool)
    v = _basename(lhs)
    if !(v in st.stoch)
        delete!(st.det, v)           # a container allocated earlier, now random
        push!(st.stoch, v)
        st.kinds[v] = _distkind(rhs)
        st.parents[v] = Set{Symbol}()
    end
    reads = _readsyms(rhs)
    lhs isa Expr && lhs.head === :ref && foreach(i -> _readsyms(i, reads), lhs.args[2:end])
    union!(st.parents[v], _resolve(st, reads, v))
    push!(st.lines, ModelLine(:tilde, lhs, rhs, dot, copy(st.loops), v))
end

function _assign!(st::_ParseState, lhs, rhs, dot::Bool)
    targets = lhs isa Expr && lhs.head === :tuple ? lhs.args : (lhs,)
    deps = _resolve(st, _readsyms(rhs))
    for t in targets
        t isa Symbol || (t isa Expr && t.head === :ref) || continue
        b = _basename(t)
        b in st.stoch && continue
        prior = t isa Symbol ? Set{Symbol}() : get(st.det, b, Set{Symbol}())
        st.det[b] = union(prior, deps)
    end
    push!(st.lines, ModelLine(:assign, lhs, rhs, dot, copy(st.loops), _basename(targets[1])))
end

"""
    generative(def::Expr) -> GenerativeModel

Read a model definition (`function name(args...) … end`, as given to
`@model`) into a [`GenerativeModel`](@ref). Model arguments that are `~`'d
count as observed.
"""
function generative(fdef::Expr)
    fdef.head === :macrocall && return generative(fdef.args[end])
    def = MacroTools.splitdef(fdef)
    argsyms = Symbol[]
    for a in vcat(get(def, :args, Any[]), get(def, :kwargs, Any[]))
        n = MacroTools.splitarg(a)[1]
        n isa Symbol && push!(argsyms, n)
    end
    st = _ParseState(Set(argsyms), ModelLine[], Tuple{Symbol,Any}[], Symbol[],
                     Dict{Symbol,Symbol}(), Dict{Symbol,Set{Symbol}}(), Dict{Symbol,Set{Symbol}}())
    _walk!(st, def[:body])
    isempty(st.stoch) && error("@graphical: no `~` statements found in $(get(def, :name, "the model"))")

    inputnames = Symbol[a for a in argsyms if !(a in st.stoch) && any(v -> a in st.parents[v], st.stoch)]
    rvmap = Dict{Symbol,RV}()
    for v in st.stoch
        rvmap[v] = RV(v, st.kinds[v])
    end
    for a in inputnames
        rvmap[a] = RV(a, :continuous)
    end
    pa = Dict{RV,Vector{RV}}(rvmap[v] => RV[rvmap[p] for p in sort!(collect(st.parents[v]); by=p -> _order(p, argsyms, st.stoch))]
                            for v in st.stoch)
    g = DAG(RV[[rvmap[a] for a in inputnames]; [rvmap[v] for v in st.stoch]], pa)
    observed = RV[rvmap[v] for v in st.stoch if v in st.args]
    latent = RV[rvmap[v] for v in st.stoch if !(v in st.args)]
    name = get(def, :name, :model)
    name isa Symbol || (name = Symbol(string(name)))
    return GenerativeModel(name, argsyms, st.lines, g, observed, latent, RV[rvmap[a] for a in inputnames])
end
# Parents: stochastic variables in the order declared, then model inputs.
_order(s, args, stoch) = (i = findfirst(==(s), stoch)) === nothing ? 10_000 + something(findfirst(==(s), args), 0) : i

"""
    observe(gm, names::Symbol...) -> GenerativeModel
    unobserve(gm, names::Symbol...) -> GenerativeModel

The same model with the `~` variables `names` treated as data (or as
parameters).
"""
function observe(gm::GenerativeModel, names::Symbol...)
    all_ = vcat(gm.latent, gm.observed)
    obs = RV[v for v in all_ if v in gm.observed || v.name in names]
    lat = RV[v for v in all_ if !(v in obs)]
    return _with(gm, _stochorder(gm, obs), _stochorder(gm, lat))
end
function unobserve(gm::GenerativeModel, names::Symbol...)
    all_ = vcat(gm.latent, gm.observed)
    lat = RV[v for v in all_ if v in gm.latent || v.name in names]
    obs = RV[v for v in all_ if !(v in lat)]
    return _with(gm, _stochorder(gm, obs), _stochorder(gm, lat))
end
_with(gm, obs, lat) = GenerativeModel(gm.name, gm.args, gm.lines, gm.dag, obs, lat, gm.inputs)
_stochorder(gm, vs) = RV[v for v in topological_order(gm.dag) if v in vs]

const _REGISTRY = IdDict{Any,GenerativeModel}()

"""
    register!(f, gm::GenerativeModel)

Associate a model function (or evaluator) with its graphical model, so
`generative(f)` finds it. `@graphical` does this for you.
"""
register!(f, gm::GenerativeModel) = (_REGISTRY[f] = gm; f)

"""
    generative(f) -> GenerativeModel

The graphical model registered for the model function `f` by
[`@graphical`](@ref). With PracticalBayes loaded, also accepts a
`PracticalBayes.Model` (an instantiated, possibly conditioned model), in which
case data are the arguments that are not `missing` plus anything conditioned on.
"""
function generative(f)
    haskey(_REGISTRY, f) && return _REGISTRY[f]
    throw(ArgumentError("no graphical model registered for $(f); define it with `@graphical @model function …`"))
end
generative(gm::GenerativeModel) = gm

"""
    @graphical @model function name(args...) … end

Define a model exactly as the inner macro would (PracticalBayes' or Turing's
`@model`), and also read its `~` statements into a [`GenerativeModel`](@ref),
retrievable with `generative(name)` or, with PracticalBayes,
`generative(name(data...))`.
"""
macro graphical(ex)
    gm = generative(ex)
    name = gm.name
    evalname = Symbol("_", name, "_eval")
    isdef = Expr(:macrocall, Symbol("@isdefined"), __source__, evalname)
    return quote
        $(esc(ex))
        $register!($(esc(name)), $gm)
        if $(esc(isdef))
            $register!($(esc(evalname)), $gm)
        end
        $(esc(name))
    end
end

"""
    @generative function name(args...) … end

Read a model definition into a [`GenerativeModel`](@ref) without defining
anything.
"""
macro generative(ex)
    return generative(ex)
end

# ---------------------------------------------------------------------------
# Probability expressions from a generative model
# ---------------------------------------------------------------------------

dag(gm::GenerativeModel) = gm.dag

"""
    factorize(gm::GenerativeModel; name = :P, plates = true) -> PExpr

The joint density of the model's `~` variables given its inputs, as the product
of each variable given its parents. `name` names the distribution (`:π` gives
`π(α) π(β) …`). With `plates = true`, a `~` inside a loop becomes a product
over the loop index, `∏_{j=1}^{J} p(θ[j] | μ, τ)`, with its parents indexed
where the code indexes them (`y[j] | θ[j]`, `x[t] | x[t-1]`); with
`plates = false` every variable is one node, `p(θ | μ, τ)`.
"""
factorize(gm::GenerativeModel; name::Symbol=:P, plates::Bool=true) =
    plates ? _plate_factorize(gm, name) : factorize(gm.dag; exclude=gm.inputs, name)

"""
    posterior(gm; normalized = false, density = :P, plates = true) -> Equation

Bayes' theorem for the model: `p(latent | observed, inputs) ∝ ∏ p(v | pa(v))`,
or with `normalized = true` the same divided by its integral over the latent
variables. `density` names the distribution, e.g. `density = :π` for
`π(θ | y) ∝ π(y | θ) π(θ)`. Accepts anything [`generative`](@ref) does.
"""
function posterior(gm::GenerativeModel; normalized::Bool=false, density::Symbol=:P, plates::Bool=true)
    isempty(gm.latent) && throw(ArgumentError("the model has no latent variables"))
    lhs = Prob(Atom[gm.latent...], Atom[gm.observed..., gm.inputs...], density)
    F = factorize(gm; name=density, plates)
    normalized || return Equation(lhs, F; relation=:propto)
    return Equation(lhs, divide(F, marg(gm.latent, F)))
end
posterior(x; kwargs...) = posterior(generative(x); kwargs...)

"""
    prior(gm) -> PExpr

The prior: the factors of the latent variables only, after integrating out the
observed ones.
"""
prior(gm::GenerativeModel) = simplify(marg(gm.observed, factorize(gm; plates=false)); combine=false)

"""
    likelihood(gm) -> PExpr

The factors of the observed variables.
"""
function likelihood(gm::GenerativeModel)
    F = _factors(factorize(gm; plates=false))
    return mul((f for f in F if f isa Prob && any(v -> v in gm.observed, _rvsof(f.targets)))...)
end

dag(x) = dag(generative(x))

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

function _role(gm::GenerativeModel, v::RV)
    v in gm.observed && return "observed"
    v in gm.latent && return "latent"
    return "input"
end

# Strip broadcasting dots so `Normal.(a .+ b .* x, σ)` reads `Normal(a + b x, σ)`.
_undot(x) = x
function _undot(ex::Expr)
    if ex.head === :. && length(ex.args) == 2 && ex.args[2] isa Expr && ex.args[2].head === :tuple
        return Expr(:call, _undot(ex.args[1]), map(_undot, ex.args[2].args)...)
    elseif ex.head === :call && ex.args[1] isa Symbol && startswith(string(ex.args[1]), ".") &&
           length(string(ex.args[1])) > 1
        return Expr(:call, Symbol(string(ex.args[1])[2:end]), map(_undot, ex.args[2:end])...)
    elseif ex.head === :macrocall && ex.args[1] === Symbol("@.")
        return _undot(ex.args[end])
    end
    return Expr(ex.head, map(_undot, ex.args)...)
end

function _table(gm::GenerativeModel, backend)
    order = topological_order(gm.dag)
    data = Matrix{Any}(undef, length(order), 4)
    for (i, v) in enumerate(order)
        data[i, 1] = backend in (:latex, :markdown) ? latex(v) : string(v.name)
        data[i, 2] = _role(gm, v)
        lines = [l for l in gm.lines if l.kind === :tilde && l.node === v.name]
        data[i, 3] = isempty(lines) ? "" : join((backend in (:latex, :markdown) ?
                                                 "\$" * _lxcode(_undot(l.rhs)) * "\$" : string(_undot(l.rhs)) for l in lines), "; ")
        backend in (:latex, :markdown) && (data[i, 3] = LaTeXString(data[i, 3]))
        data[i, 4] = join((p.name for p in gm.dag.parents[v]), ", ")
    end
    return (data=data, labels=Any["variable", "role", "distribution", "parents"], title="",
            alignment=[:r, :l, :l, :l])
end

tabulate(io::IO, gm::GenerativeModel; backend::Symbol=:text, kwargs...) =
    _pt(io, _table(gm, backend), backend; kwargs...)

Base.show(io::IO, gm::GenerativeModel) =
    print(io, "GenerativeModel ", gm.name, "(", join(gm.args, ", "), ")")
function Base.show(io::IO, ::MIME"text/plain", gm::GenerativeModel)
    show(io, gm)
    println(io)
    tabulate(io, gm)
    if !isempty(gm.latent)
        print(io, "\n  posterior: ")
        show(io, posterior(gm))
    end
end

# `x[i]`, `θ[j, k]` with plain indices become `x_i`, `θ_j,k`, which Latexify
# subscripts uniformly (it renders `theta[j]` as a call otherwise).
# Latexify turns a symbol spelled `theta` into θ, but not the start of
# `theta_j`, so Greek names are mapped to their letters first.
const _GREEK = Dict(
    "alpha" => "α", "beta" => "β", "gamma" => "γ", "delta" => "δ", "epsilon" => "ϵ",
    "zeta" => "ζ", "eta" => "η", "theta" => "θ", "iota" => "ι", "kappa" => "κ",
    "lambda" => "λ", "mu" => "μ", "nu" => "ν", "xi" => "ξ", "pi" => "π", "rho" => "ρ",
    "sigma" => "σ", "tau" => "τ", "upsilon" => "υ", "phi" => "ϕ", "chi" => "χ",
    "psi" => "ψ", "omega" => "ω", "Gamma" => "Γ", "Delta" => "Δ", "Theta" => "Θ",
    "Lambda" => "Λ", "Xi" => "Ξ", "Pi" => "Π", "Sigma" => "Σ", "Phi" => "Φ",
    "Psi" => "Ψ", "Omega" => "Ω")
_refsub(x) = x
function _refsub(ex::Expr)
    if ex.head === :ref && ex.args[1] isa Symbol && all(i -> i isa Union{Symbol,Integer}, ex.args[2:end])
        base = string(ex.args[1])
        return Symbol(string(get(_GREEK, base, base), "_", join(ex.args[2:end], ",")))
    end
    return Expr(ex.head, map(_refsub, ex.args)...)
end

# Model code can contain anything (type parameters, closures); Latexify does
# not render all of it, so fall back to typewriter text rather than fail.
function _lxcode(x)
    try
        return _lx(_refsub(x))
    catch
        s = replace(string(x), "\\" => "\\textbackslash{}", "_" => "\\_", "{" => "\\{", "}" => "\\}",
                    "\$" => "\\\$", "&" => "\\&", "#" => "\\#", "%" => "\\%")
        return string("\\texttt{", s, "}")
    end
end

function _lxline(l::ModelLine)
    lhs = _lxcode(l.lhs)
    rhs = _lxcode(_undot(l.rhs))
    op = l.kind === :tilde ? "\\sim" : "="
    s = string(lhs, " &", op, " ", rhs)
    if !isempty(l.loops)
        s *= " && " * join((_lxloop(i, r) for (i, r) in l.loops), ",\\; ")
    elseif l.dot
        s *= " && \\text{elementwise}"
    end
    return s
end
function _lxloop(i, r)
    if r isa Expr && r.head === :call && r.args[1] === :(:) && length(r.args) == 3
        return string(_lxcode(i), " = ", _lxcode(r.args[2]), ", \\ldots, ", _lxcode(r.args[3]))
    end
    return string(_lxcode(i), " \\in ", _lxcode(r))
end

function _latex(gm::GenerativeModel; posterior::Bool=true)
    # An assignment to a variable that is later `~`'d only allocates its
    # container (`theta = Vector{…}(undef, J)`): not part of the model.
    stoch = Set(v.name for v in vcat(gm.latent, gm.observed))
    rows = String[_lxline(l) for l in gm.lines if l.kind === :tilde || !(l.node in stoch)]
    s = join(rows, " \\\\\n")
    if posterior && !isempty(gm.latent)
        eq = PracticalProbability.posterior(gm)
        s *= string(" \\\\[6pt]\n", _latex(eq.lhs), " &\\propto ", _latex(eq.rhs))
    end
    return string("\\begin{aligned}\n", s, "\n\\end{aligned}")
end
latex(gm::GenerativeModel; env::Symbol=:inline, posterior::Bool=true) = _wrap(_latex(gm; posterior), env)
# Anything `generative` understands (a registered model function, or with
# PracticalBayes loaded a `Model`) renders as its graphical model.
_latex(x) = _latex(generative(x))
tabulate(io::IO, x; kwargs...) = tabulate(io, generative(x); kwargs...)
Base.show(io::IO, ::MIME"text/latex", gm::GenerativeModel) = print(io, "\$\$", _latex(gm), "\$\$")
Latexify.latexify(gm::GenerativeModel; env::Symbol=:inline, kwargs...) = latex(gm; env)

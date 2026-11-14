# Numbers: factors, discrete Bayesian networks, and exact computation of any
# probability expression against a network. This is also what checks the
# symbolic rules: an identity that holds must compute equal on every network.

"""
    Factor

A function of discrete random variables, stored as an array with one dimension
per variable. The result of [`compute`](@ref) whenever the expression still
has free variables. Implements the Tables.jl interface (one row per
assignment), so it prints as a table and converts to a `DataFrame`.

Index it with values: `f[A = 1, B = 2]` or `f(A => 1, B => 2)`.
"""
struct Factor
    vars::Vector{RV}
    domains::Vector{Vector{Any}}
    values::Array{Float64}
    label::String
end
Factor(vars, domains, values) = Factor(vars, domains, values, "")
_scalar_factor(x::Real, label="") = Factor(RV[], Vector{Any}[], fill(Float64(x)), label)

_relabel(f::Factor, label) = Factor(f.vars, f.domains, f.values, label)

function _align(f::Factor, vars::Vector{RV})
    # Permute f's array to the order its variables take in `vars`, then
    # reshape with singleton dimensions for the variables f lacks.
    pos = [findfirst(==(v), vars) for v in f.vars]
    perm = sortperm(pos)
    A = isempty(perm) ? f.values : permutedims(f.values, perm)
    sz = ones(Int, length(vars))
    for (k, i) in enumerate(pos[perm])
        sz[i] = size(A, k)
    end
    return reshape(A, Tuple(sz))
end

function _combine(op, f::Factor, g::Factor)
    vars = union(f.vars, g.vars)
    doms = Vector{Any}[]
    for v in vars
        i = findfirst(==(v), f.vars)
        j = findfirst(==(v), g.vars)
        if i !== nothing && j !== nothing && f.domains[i] != g.domains[j]
            throw(ArgumentError("variable $(v.name) has different domains in the two factors"))
        end
        push!(doms, i === nothing ? g.domains[j] : f.domains[i])
    end
    vals = broadcast(op, _align(f, vars), _align(g, vars))
    vals isa Array || (vals = fill(vals))
    return Factor(vars, doms, vals)
end

Base.:*(f::Factor, g::Factor) = _combine(*, f, g)
Base.:/(f::Factor, g::Factor) = _combine(/, f, g)
Base.:+(f::Factor, g::Factor) = _combine(+, f, g)
Base.:-(f::Factor, g::Factor) = _combine(-, f, g)

"""
    sumout(f, vars) -> Factor

Sum the factor over the given variables.
"""
function sumout(f::Factor, vs)
    vs = _rvlist(vs)
    dims = Tuple(i for (i, v) in enumerate(f.vars) if v in vs)
    isempty(dims) && return f
    keep = [i for i in eachindex(f.vars) if !(i in dims)]
    vals = dropdims(sum(f.values; dims=dims); dims=dims)
    vals isa Array || (vals = fill(vals))
    return Factor(f.vars[keep], f.domains[keep], vals)
end

function _value_index(f::Factor, assignment)
    length(f.vars) == length(assignment) ||
        throw(ArgumentError("give a value for each of $(join((v.name for v in f.vars), ", "))"))
    idx = map(f.vars, f.domains) do v, d
        key = findfirst(a -> a[1] == v.name || (a[1] isa RV && a[1] == v), assignment)
        key === nothing && throw(ArgumentError("no value given for $(v.name)"))
        val = assignment[key][2]
        k = findfirst(x -> isequal(x, val), d)
        k === nothing && throw(ArgumentError("$(repr(val)) is not a value of $(v.name)"))
        k
    end
    return f.values[idx...]
end
Base.getindex(f::Factor; kw...) = _value_index(f, [(k, v) for (k, v) in kw])
(f::Factor)(ps::Pair...) = _value_index(f, [(p.first, p.second) for p in ps])

Base.Float64(f::Factor) = isempty(f.vars) ? f.values[] : throw(ArgumentError("factor still depends on $(f.vars)"))
Base.convert(::Type{Float64}, f::Factor) = Float64(f)

function Base.isapprox(f::Factor, g::Factor; kwargs...)
    issetequal(f.vars, g.vars) || return false
    vars = f.vars
    return isapprox(_align(f, vars), _align(g, vars); kwargs...)
end

# ---------------------------------------------------------------------------
# Bayesian networks
# ---------------------------------------------------------------------------

"""
    BayesNet(specs::Pair...; domains = Dict())

A discrete Bayesian network. Each spec gives one variable's conditional
probability table (CPT):

```julia
@rv X Y Z
bn = BayesNet(
    X => [0.3, 0.7],                    # P(X)
    (Y | X) => [0.9 0.2; 0.1 0.8],      # P(Y | X): rows are Y, columns are X
    (Z | (X, Y)) => cat([0.5 0.4; 0.5 0.6], [0.1 0.7; 0.9 0.3]; dims = 3),
)
```

The first array dimension is the variable, the rest are its parents in the
order written; every column must sum to one. Values are `1:n` unless given in
`domains`, e.g. `domains = Dict(X => [:no, :yes])`. The graph is read off the
specs.

A named event (`@event A`) is a yes/no variable with values `[true, false]`,
and its table may give just the probability that it happens: `A => 0.3` is
`P(A) = 0.3`, and `(B | A) => [0.9, 0.2]` is `P(B | A) = 0.9`,
`P(B | Aᶜ) = 0.2`.
"""
struct BayesNet
    dag::DAG
    cpts::Dict{RV,Factor}
end

const _EVENT_DOMAIN = Any[true, false]

function BayesNet(specs::Pair...; domains::AbstractDict=Dict{RV,Any}(), atol::Real=1e-8)
    doms = Dict{RV,Vector{Any}}(k => collect(Any, v) for (k, v) in domains)
    raw = Tuple{RV,Vector{RV},Array{Float64}}[]
    for (lhs, tbl) in specs
        if lhs isa RV
            v, ps = lhs, RV[]
        elseif lhs isa Given && length(lhs.targets) == 1 && all(a -> a isa RV, lhs.given) && lhs.targets[1] isa RV
            v, ps = lhs.targets[1], RV[a for a in lhs.given]
        else
            throw(ArgumentError("a CPT is written `A => table` or `(A | B, C) => table`, got $(lhs)"))
        end
        T = tbl isa Real ? fill(Float64(tbl)) : Array{Float64}(tbl)
        # an event's table may give only P(event happens | parents)
        if isevent(v) && ndims(T) == length(ps)
            # `map`, not `1 .- T`: broadcasting over a 0-dim array returns a scalar
            T = cat(reshape(T, 1, size(T)...), reshape(map(x -> 1 - x, T), 1, size(T)...); dims=1)
        end
        ndims(T) == 1 + length(ps) ||
            throw(ArgumentError("table for $(v.name) needs $(1 + length(ps)) dimensions, has $(ndims(T))"))
        push!(raw, (v, ps, T))
        for (k, x) in enumerate(vcat(v, ps))
            n = size(T, k)
            if haskey(doms, x)
                length(doms[x]) == n ||
                    throw(ArgumentError("$(x.name) has $(length(doms[x])) values but a table dimension of size $n"))
            elseif isevent(x)
                n == 2 || throw(ArgumentError("event $(x.name) has two values (happens, doesn't), got $n"))
                doms[x] = copy(_EVENT_DOMAIN)
            else
                doms[x] = collect(Any, 1:n)
            end
        end
    end
    cpts = Dict{RV,Factor}()
    pa = Dict{RV,Vector{RV}}()
    for (v, ps, T) in raw
        haskey(cpts, v) && throw(ArgumentError("two tables for $(v.name)"))
        sums = sum(T; dims=1)
        all(s -> isapprox(s, 1; atol), sums) ||
            throw(ArgumentError("the table for $(v.name) must sum to one over $(v.name) for every parent value"))
        all(>=(0), T) || throw(ArgumentError("the table for $(v.name) has negative entries"))
        cpts[v] = Factor(vcat(v, ps), [doms[x] for x in vcat(v, ps)], T,
                         sprint(show, Prob(Atom[v], Atom[ps...])))
        pa[v] = ps
    end
    for (v, ps) in pa, p in ps
        haskey(cpts, p) || throw(ArgumentError("$(p.name) is a parent of $(v.name) but has no table"))
    end
    order = RV[v for (v, _, _) in raw]
    return BayesNet(DAG(order, pa), cpts)
end

"""
    rand_bayesnet(g::DAG; card = 2, rng = Random.default_rng()) -> BayesNet

A network on the graph `g` with random CPTs; every variable takes values
`1:card` (events: `true`/`false`). Useful for checking a symbolic identity
numerically.
"""
function rand_bayesnet(g::DAG; card::Integer=2, rng::AbstractRNG=Random.default_rng())
    specs = Pair[]
    n(v) = isevent(v) ? 2 : card
    for v in topological_order(g)
        ps = g.parents[v]
        T = rand(rng, n(v), map(n, ps)...) .+ 0.05
        T ./= sum(T; dims=1)
        push!(specs, (isempty(ps) ? v : Given(Atom[v], Atom[ps...])) => T)
    end
    return BayesNet(specs...)
end

dag(bn::BayesNet) = bn.dag

"""
    cpt(bn, v) -> Factor

The conditional probability table of `v` as a [`Factor`](@ref).
"""
cpt(bn::BayesNet, v::RV) = bn.cpts[v]

# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

# Networks for each named distribution: one network for every name, or a Dict
# name => network (with :P as the fallback).
const _Nets = Union{BayesNet,AbstractDict{Symbol,BayesNet}}
_net(bn::BayesNet, ::Symbol) = bn
function _net(nets::AbstractDict{Symbol,BayesNet}, name::Symbol)
    haskey(nets, name) && return nets[name]
    haskey(nets, :P) && return nets[:P]
    throw(ArgumentError("no network for the distribution $(name); pass Dict(:$(name) => bn, …)"))
end
_anynet(bn::BayesNet) = bn
_anynet(nets::AbstractDict{Symbol,BayesNet}) = haskey(nets, :P) ? nets[:P] : first(values(nets))

"""
    compute(expr, bn::BayesNet) -> Float64 or Factor
    compute(expr, nets::AbstractDict{Symbol,BayesNet})

The value of `expr` under the network `bn`, computed exactly by enumeration.
The result is a number if `expr` has no free variables, otherwise a
[`Factor`](@ref) over them. `Do(⋅)` terms are computed in the network with the
intervened variables' tables removed (the truncated factorisation). Events of
any shape (`A ∪ B`, `X(1)'`, …) are computed through their indicator
functions.

With a `Dict`, each term `π(⋅)` is computed in `nets[:π]` (falling back to
`nets[:P]`), so expressions mixing distributions, such as importance weights
`p(x) / q(x)`, can be checked.
"""
function compute(e::PExpr, nets::_Nets)
    f = _eval(e, nets)
    isempty(f.vars) && return f.values[]
    return _relabel(_permute(f, _display_order(e)), sprint(show, e))
end
compute(eq::Equation, nets::_Nets) = (compute(eq.lhs, nets), compute(eq.rhs, nets))

# Targets before conditioning variables, in the order they were written.
_display_order(p::Prob) = RV[v for v in (freerv(a) for a in vcat(p.targets, p.given)) if v !== nothing]
_display_order(e::PExpr) = freevars(e)

function _permute(f::Factor, order::Vector{RV})
    vars = vcat(RV[v for v in order if v in f.vars], RV[v for v in f.vars if !(v in order)])
    perm = [findfirst(==(v), f.vars) for v in vars]
    perm == eachindex(f.vars) && return f
    return Factor(vars, f.domains[perm], permutedims(f.values, perm), f.label)
end

_eval(l::Literal, ::_Nets) = _scalar_factor(l.value)
_eval(p::Product, nets::_Nets) = reduce(*, (_eval(f, nets) for f in p.factors))
_eval(s::Sum, nets::_Nets) = reduce(+, (_eval(t, nets) for t in s.terms))
_eval(q::Quotient, nets::_Nets) = _eval(q.num, nets) / _eval(q.den, nets)
function _eval(m::Marginal, nets::_Nets)
    f = _eval(m.body, nets)
    for v in m.over
        if v in f.vars
            f = sumout(f, v)
        else     # summing a constant over a variable multiplies by its domain size
            f = Factor(f.vars, f.domains, f.values .* length(domain(_anynet(nets), v)))
        end
    end
    return f
end

# Whether an event holds for the values `vals` of its variables.
_holds(x::RV, vals) = vals[x] == true
_holds(e::Event, vals) = isequal(vals[e.rv], e.value)
_holds(c::Complement, vals) = !_holds(c.event, vals)
_holds(u::EventUnion, vals) = any(x -> _holds(x, vals), u.parts)
_holds(u::EventIntersection, vals) = all(x -> _holds(x, vals), u.parts)
_holds(::SureEvent, vals) = true
_holds(::ImpossibleEvent, vals) = false

# Every `X = x` inside an event must name a value X can take.
_checkvalues(e::Event, bn) = any(x -> isequal(x, e.value), domain(bn, e.rv)) ||
    throw(ArgumentError("$(repr(e.value)) is not a value of $(e.rv.name); its domain is $(domain(bn, e.rv))"))
_checkvalues(c::Complement, bn) = _checkvalues(c.event, bn)
_checkvalues(u::Union{EventUnion,EventIntersection}, bn) = foreach(x -> _checkvalues(x, bn), u.parts)
_checkvalues(::Atom, bn) = nothing

# The indicator of an event as a factor over its variables.
function _indicator(e::Atom, bn::BayesNet)
    _checkvalues(e, bn)
    vs = rvs(e)
    isempty(vs) && return _scalar_factor(_holds(e, Dict{RV,Any}()) ? 1.0 : 0.0)
    doms = Vector{Any}[domain(bn, v) for v in vs]
    vals = Array{Float64}(undef, map(length, doms)...)
    for idx in CartesianIndices(vals)
        assignment = Dict{RV,Any}(v => doms[k][idx[k]] for (k, v) in enumerate(vs))
        vals[idx] = _holds(e, assignment) ? 1.0 : 0.0
    end
    return Factor(vs, doms, vals)
end

function _eval(p::Prob, nets::_Nets)
    bn = _net(nets, p.name)
    _hascf(p) && return _cf_eval(p, bn)        # counterfactual.jl
    p, back = _resolve_copies(p, bn)
    free = RV[v for v in (freerv(a) for a in p.targets) if v !== nothing]
    evs = Atom[a for a in p.targets if freerv(a) === nothing]
    f = _cond_factor(bn, p.name, free, evs, p.given)
    isempty(back) && return f
    return Factor(RV[get(back, v, v) for v in f.vars], f.domains, f.values, f.label)
end

# A primed variable `x′` (as the ID algorithm writes a summed copy of `x`) is
# another value of `x`: a term that mentions `x′` but not `x` is computed with
# `x` and the result relabelled.
_unprimed(v::RV) = (s = string(v.name); endswith(s, "′") ? RV(Symbol(rstrip(s, '′')), v.kind) : nothing)
function _resolve_copies(p::Prob, bn::BayesNet)
    vs = _rvsof(vcat(p.targets, p.given))
    d = Dict{Atom,Atom}()
    back = Dict{RV,RV}()
    for v in vs
        v in bn.dag.nodes && continue
        b = _unprimed(v)
        (b === nothing || !(b in bn.dag.nodes)) && continue
        b in vs && throw(ArgumentError("$(p) mentions both $(b.name) and its copy $(v.name)"))
        d[v] = b
        back[b] = v
    end
    isempty(d) && return (p, back)
    return (_subs(p, d), back)
end
domain(bn::BayesNet, v::RV) = haskey(bn.cpts, v) ? bn.cpts[v].domains[1] :
    (b = _unprimed(v); b !== nothing && haskey(bn.cpts, b) ? bn.cpts[b].domains[1] : bn.cpts[v].domains[1])

# P(free, events | given) as a factor over `free` and the free given variables.
# `free` may include event variables, to get their whole distribution (over
# true/false) rather than the probability that they happen.
function _cond_factor(bn::BayesNet, name::Symbol, free::Vector{RV}, targetev::Vector{Atom}, given::Vector{Atom})
    for v in vcat(free, _rvsof(vcat(targetev, given)))
        v in bn.dag.nodes || throw(ArgumentError("$(v.name) is not in the network for $(name)"))
    end
    X = Set(rv(a) for a in given if a isa Intervention)
    J = reduce(*, (bn.cpts[v] for v in topological_order(bn.dag) if !(v in X)); init=_scalar_factor(1.0))
    # Events (anything but a free variable) enter as indicator factors.
    ev(a) = a isa Intervention ? a.target : a
    givenev = [ev(a) for a in given if freerv(ev(a)) === nothing]
    keep_den = Set(freerv(ev(a)) for a in given if freerv(ev(a)) !== nothing)
    keep_num = union(Set(free), keep_den)
    den = reduce(*, (_indicator(e, bn) for e in givenev); init=J)
    num = reduce(*, (_indicator(e, bn) for e in targetev); init=den)
    num = sumout(num, [v for v in num.vars if !(v in keep_num)])
    den = sumout(den, [v for v in den.vars if !(v in keep_den)])
    out = num / den
    # A free do(X) whose X affects nothing is still an argument of the result.
    for v in keep_num
        if !(v in out.vars)
            d = domain(bn, v)
            out = out * Factor(RV[v], Vector{Any}[d], ones(length(d)))
        end
    end
    return out
end

"""
    verify(eq, nets; atol = 1e-10) -> Bool
    verify(d::Derivation, nets; atol)
    verify(lhs, rhs, nets; atol)

Check numerically that both sides of an equation (or every line of a
derivation) agree under `nets`: one [`BayesNet`](@ref), or a `Dict` from
distribution names to networks.
"""
function verify(eq::Equation, nets::_Nets; atol::Real=1e-10)
    eq.relation === :eq || throw(ArgumentError("only equalities can be checked numerically"))
    return _approx(compute(eq.lhs, nets), compute(eq.rhs, nets); atol)
end
verify(d::Derivation, nets::_Nets; atol::Real=1e-10) =
    all(e -> _approx(compute(first(d), nets), compute(e, nets); atol), d.exprs)
verify(a::PExpr, b::PExpr, nets::_Nets; atol::Real=1e-10) = verify(Equation(a, b), nets; atol)

# Equal as functions: a factor that does not mention a variable is constant in it.
_approx(a::Real, b::Real; atol) = isapprox(a, b; atol)
_approx(a::Real, b::Factor; atol) = _approx(_scalar_factor(a), b; atol)
_approx(a::Factor, b::Real; atol) = _approx(a, _scalar_factor(b); atol)
function _approx(a::Factor, b::Factor; atol)
    vars = union(a.vars, b.vars)
    return all(broadcast((x, y) -> isapprox(x, y; atol) || (isnan(x) && isnan(y)),
                         _align(a, vars), _align(b, vars)))
end

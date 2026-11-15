# Linear programming, exact for rationals: a dense two-phase simplex with
# Bland's rule (no cycling). It gives `deduce` its bounds when the facts leave
# a probability undetermined, and `lp_bounds` the sharp bounds of Balke &
# Pearl (1997) on causal effects that are not identified.

# min c'x subject to A x = b, x ≥ 0. Returns (:optimal, value, x),
# (:infeasible, nothing, nothing) or (:unbounded, nothing, nothing).
function _simplex(c::AbstractVector, A::AbstractMatrix, b::AbstractVector)
    # exact when every entry is (abstract element types such as Real can still hold only rationals)
    exact = all(x -> x isa _Exact, c) && all(x -> x isa _Exact, A) && all(x -> x isa _Exact, b)
    T = exact ? Rational{BigInt} : Float64

    A = T.(A); b = T.(b); c = T.(c)
    # independent rows only
    R, piv = _rref(hcat(A, b))
    any(==(size(A, 2) + 1), piv) && return (:infeasible, nothing, nothing)
    A, b = R[1:length(piv), 1:end-1], R[1:length(piv), end]
    m, n = size(A)
    for i in 1:m
        if b[i] < 0
            A[i, :] .= .-A[i, :]; b[i] = -b[i]
        end
    end
    tol = T <: Rational ? zero(T) : T(1e-10)
    # phase 1: artificials n+1 … n+m
    tab = hcat(A, Matrix{T}(_eye(m)), b)
    basis = collect(n+1:n+m)
    obj1 = vcat(zeros(T, n), ones(T, m))
    _run_simplex!(tab, basis, obj1, tol) || return (:unbounded, nothing, nothing)   # cannot happen in phase 1
    if _objective(tab, basis, obj1) > tol
        return (:infeasible, nothing, nothing)
    end
    # drive artificials out of the basis
    for i in 1:m
        basis[i] > n || continue
        for j in 1:n
            if abs(tab[i, j]) > tol
                _pivot!(tab, basis, i, j)
                break
            end
        end
    end
    tab2 = hcat(tab[:, 1:n], tab[:, end])
    _run_simplex!(tab2, basis, c, tol) || return (:unbounded, nothing, nothing)
    x = zeros(T, n)
    for (i, j) in enumerate(basis)
        x[j] = tab2[i, end]
    end
    return (:optimal, dot(c, x), x)
end

_objective(tab, basis, obj) = sum(obj[basis[i]] * tab[i, end] for i in eachindex(basis))

function _pivot!(tab, basis, r, s)
    tab[r, :] ./= tab[r, s]
    for i in axes(tab, 1)
        i == r && continue
        f = tab[i, s]
        iszero(f) || (tab[i, :] .-= f .* tab[r, :])
    end
    basis[r] = s
end

# Simplex iterations on a tableau in canonical form for `basis`. False if unbounded.
function _run_simplex!(tab, basis, obj, tol)
    m = size(tab, 1)
    n = size(tab, 2) - 1
    for _ in 1:100_000
        # reduced costs
        y = [obj[basis[i]] for i in 1:m]
        s = nothing
        for j in 1:n
            j in basis && continue
            rc = obj[j] - sum(y[i] * tab[i, j] for i in 1:m)
            if rc < -tol
                s = j                     # Bland: the first improving column
                break
            end
        end
        s === nothing && return true
        r = nothing
        best = nothing
        for i in 1:m
            tab[i, s] > tol || continue
            ratio = tab[i, end] / tab[i, s]
            if best === nothing || ratio < best - tol || (abs(ratio - best) <= tol && basis[i] < basis[r])
                best, r = ratio, i
            end
        end
        r === nothing && return false
        _pivot!(tab, basis, r, s)
    end
    throw(ErrorException("the simplex method did not converge"))
end

# min and max of c'x over {A x = b, x ≥ 0}.
function _lp_range(c, A, b)
    lo = _simplex(c, A, b)
    lo[1] === :infeasible && throw(ArgumentError("the constraints are inconsistent (no probabilities satisfy them)"))
    hi = _simplex(.-c, A, b)
    (lo[1] === :optimal && hi[1] === :optimal) || throw(ArgumentError("the bounds are infinite"))
    return (_shrink(something(lo[2])), _shrink(-something(hi[2])))
end

# ---------------------------------------------------------------------------
# Bounds in deduce
# ---------------------------------------------------------------------------

# Bounds on an undetermined query over the cell probabilities: a linear form
# directly, a conditional probability P(T | G) by the Charnes–Cooper
# transformation (y = t p, t = 1 / P(G)). `nothing` for other shapes.
function _deduce_bounds(e::PExpr, S::_EventSpace)
    A, b = S.A, S.b
    n = size(A, 2)
    if e isa Prob && !isempty(e.given)
        num = _cellvec(S, vcat(e.targets, e.given))
        den = _cellvec(S, e.given)
        # variables (y, t): A y - b t = 0, den ⋅ y = 1, y, t ≥ 0
        A2 = vcat(hcat(A, -b), hcat(permutedims(den), zeros(Int, 1, 1)))
        b2 = vcat(zeros(Int, size(A, 1)), 1)
        return _lp_range(vcat(num, 0), A2, b2)
    end
    c = try
        _linform(S, e)
    catch
        return nothing
    end
    e isa Literal && return nothing
    return _lp_range(c, A, b)
end

# ---------------------------------------------------------------------------
# Causal bounds by linear programming (Balke & Pearl)
# ---------------------------------------------------------------------------

"""
    LPBounds

`lower ≤ query ≤ upper`: the sharp bounds of [`lp_bounds`](@ref).
"""
struct LPBounds
    query::PExpr
    lower::Real
    upper::Real
end
Base.show(io::IO, b::LPBounds) = print(io, _numstr(b.lower), " ≤ ", b.query, " ≤ ", _numstr(b.upper))
Base.Tuple(b::LPBounds) = (b.lower, b.upper)
Base.extrema(b::LPBounds) = (b.lower, b.upper)

"""
    lp_bounds(query, g::ADMG, data; maxtypes = 50_000) -> LPBounds

Bounds on a causal quantity that the observed distribution does not
identify, by linear programming over response functions (Balke & Pearl
1997). `query` is an interventional probability such as
`P(Y(1) | Do(X(1)))`, or a linear combination of them such as the average
causal effect `P(Y(1) | Do(X(1))) - P(Y(1) | Do(X(0)))`. `data` is a
[`BayesNet`](@ref) whose distribution of the observed variables of `g` is the
observed one (it may contain hidden variables too; they are summed out).

The variables of `g` without parents and without bidirected edges are
exogenous (instruments, randomised treatments); the others get a response
function each (a value for every configuration of their observed parents),
with an arbitrary joint distribution. The bounds are the minimum and maximum
of the query over all such models that reproduce the observed distribution.
They are sharp when the non-exogenous variables form one district, as in the
instrumental-variable model `Z → X → Y, X ↔ Y`; with less confounding than
that they are valid but may be wider than necessary. Exact for rational data.
"""
function lp_bounds(query::PExpr, g::Union{ADMG,DAG}, data::BayesNet; maxtypes::Integer=50_000)
    G = _asadmg(g)
    order = topological_order(G)
    sibs = Set(v for e in G.bidirected for v in e)
    W = RV[v for v in order if isempty(parents(G, v)) && !(v in sibs)]
    Rv = RV[v for v in order if !(v in W)]
    isempty(Rv) && throw(ArgumentError("every variable is exogenous: the effect is identified"))
    dom = Dict(v => domain(data, v) for v in order)
    # response functions: one value per configuration of the observed parents
    cfgs = Dict(v => vec(collect(Iterators.product((dom[p] for p in parents(G, v))...))) for v in Rv)
    nf = Dict(v => length(dom[v])^length(cfgs[v]) for v in Rv)
    ntypes = prod(big(nf[v]) for v in Rv)
    ntypes > maxtypes && throw(ArgumentError(string(ntypes, " response types: too many (raise maxtypes)")))
    ntypes = Int(ntypes)
    # type k ↦ the value of v for each parent configuration
    function response(k, v)
        idx = k - 1
        for u in Rv
            u == v && break
            idx ÷= nf[u]
        end
        code = idx % nf[v]
        d = length(dom[v])
        return [dom[v][(code ÷ d^(j - 1)) % d + 1] for j in eachindex(cfgs[v])]
    end
    resp = [Dict(v => response(k, v) for v in Rv) for k in 1:ntypes]
    function run(k, vals::Dict{RV,Any}, fixed::Dict{RV,Any})
        out = copy(vals)
        for v in Rv
            if haskey(fixed, v)
                out[v] = fixed[v]
            else
                cfg = Tuple(out[p] for p in parents(G, v))
                out[v] = resp[k][v][findfirst(==(cfg), cfgs[v])]
            end
        end
        return out
    end
    # the observed distribution of R given W
    J = compute(Prob(Atom[order...], Atom[]), data)
    Jf = J isa Factor ? J : _scalar_factor(J)
    jar = _align(Jf, order)
    wconfs = vec(collect(Iterators.product((dom[w] for w in W)...)))
    rconfs = vec(collect(Iterators.product((dom[r] for r in Rv)...)))
    idx(v, x) = findfirst(y -> isequal(y, x), dom[v])
    cell(wc, rc) = (vals = Dict{RV,Any}(zip(W, wc)); merge!(vals, Dict{RV,Any}(zip(Rv, rc)));
                    jar[(idx(v, vals[v]) for v in order)...])
    pw = Dict(wc => sum(cell(wc, rc) for rc in rconfs) for wc in wconfs)
    rows = Vector{Vector{Int}}()
    rhs = Real[]
    for wc in wconfs
        iszero(pw[wc]) && continue
        wv = Dict{RV,Any}(zip(W, wc))
        obs = [Tuple(run(k, wv, Dict{RV,Any}())[r] for r in Rv) for k in 1:ntypes]
        for rc in rconfs
            push!(rows, [obs[k] == rc ? 1 : 0 for k in 1:ntypes])
            push!(rhs, _div(cell(wc, rc), pw[wc]))
        end
    end
    A = reduce(vcat, (permutedims(r) for r in rows))
    # the query as a linear function of the type probabilities
    c, offset = _lp_query(query, G, W, wconfs, pw, ntypes, run)
    lo, hi = _lp_range(c, A, map(identity, rhs))
    return LPBounds(query, _shrink(lo + offset), _shrink(hi + offset))
end

function _lp_query(q::PExpr, G, W, wconfs, pw, ntypes, run)
    terms = q isa Sum ? q.terms : PExpr[q]
    c = zeros(Real, ntypes)
    offset = 0
    total = sum(values(pw))
    for t in terms
        k, r = _coef(t)
        if r isa Literal
            offset += k * r.value
            continue
        end
        r isa Prob || throw(ArgumentError(string("lp_bounds takes interventional probabilities and their linear combinations; got ", t)))
        all(a -> a isa Intervention && a.target isa Event, r.given) ||
            throw(ArgumentError(string("lp_bounds needs every conditioning atom to be an intervention Do(X(x)); got ", r)))
        fixed = Dict{RV,Any}(a.target.rv => a.target.value for a in r.given)
        for v in _rvsof(vcat(r.targets, r.given))
            v in G.dag.nodes || throw(ArgumentError(string(v.name, " is not in the graph")))
        end
        # Exogenous variables are independent of the response types and of each
        # other; those not intervened on keep their observed (marginal) distribution.
        free = [i for (i, w) in enumerate(W) if !haskey(fixed, w)]
        fconfs = unique([Tuple(wc[i] for i in free) for wc in wconfs])
        for fc in fconfs
            p = _div(sum(pw[wc] for wc in wconfs if Tuple(wc[i] for i in free) == fc), total)
            iszero(p) && continue
            wv = Dict{RV,Any}(W[i] => fc[j] for (j, i) in enumerate(free))
            for (v, x) in fixed
                v in W && (wv[v] = x)
            end
            for j in 1:ntypes
                vals = run(j, wv, fixed)
                all(a -> _holds(a, vals), r.targets) && (c[j] += k * p)
            end
        end
    end
    return (map(identity, c), offset)
end

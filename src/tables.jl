# PrettyTables integration: every tabular object (factors, CPTs, Bayesian
# networks, derivations, independence lists, generative models) goes through
# `tabulate`, which renders it with any PrettyTables backend. Factor is also a
# Tables.jl source, so `pretty_table(f)` and `DataFrame(f)` work directly.

"""
    Independencies

The list returned by [`independencies`](@ref): a vector of
[`Independence`](@ref) statements that prints as a table.
"""
struct Independencies <: AbstractVector{Independence}
    items::Vector{Independence}
end
Base.size(x::Independencies) = size(x.items)
Base.getindex(x::Independencies, i::Int) = x.items[i]

# Cells: expressions as text for the text/HTML backends, as math for LaTeX and
# Markdown (where a bare `|` would also break the table).
_mathcell(x, backend) = backend in (:latex, :markdown) ? latex(x) : sprint(show, x)
_numcell(x::Real) = isnan(x) ? "NaN" : string(round(x; sigdigits=5))
_namecell(v::RV, backend) = backend in (:latex, :markdown) ? latex(v) : string(v.name)
_namecell(s::AbstractString, backend) = s

# All index tuples, first variable changing slowest (how tables are read).
function _assignments(f::Factor)
    rev = vec(collect(Iterators.product((eachindex(d) for d in reverse(f.domains))...)))
    return [reverse(t) for t in rev]
end

function _table(f::Factor, backend)
    rows = _assignments(f)
    data = Matrix{Any}(undef, length(rows), length(f.vars) + 1)
    for (r, idx) in enumerate(rows)
        for (k, i) in enumerate(idx)
            data[r, k] = _valstr(f.domains[k][i])
        end
        data[r, end] = _numcell(f.values[idx...])
    end
    lbl = isempty(f.label) ? "value" : f.label
    head = backend in (:latex, :markdown) && !isempty(f.label) ? LaTeXString("\$" * _flabel_latex(f) * "\$") : lbl
    labels = Any[(_namecell(v, backend) for v in f.vars)..., head]
    return (data=data, labels=labels, title="", alignment=vcat(fill(:c, length(f.vars)), :r))
end
_flabel_latex(f::Factor) = replace(f.label, "|" => "\\mid", "∑" => "\\sum", "∫" => "\\int")

# CPT in wide form: one row per parent configuration, one column per value.
function _cpt_table(f::Factor, backend)
    v = f.vars[1]
    ps = f.vars[2:end]
    pdoms = f.domains[2:end]
    rows = isempty(ps) ? [()] : vec(collect(Iterators.product((eachindex(d) for d in pdoms)...)))
    data = Matrix{Any}(undef, length(rows), length(ps) + length(f.domains[1]))
    for (r, idx) in enumerate(rows)
        for (k, i) in enumerate(idx)
            data[r, k] = _valstr(pdoms[k][i])
        end
        for (j, _) in enumerate(f.domains[1])
            data[r, length(ps)+j] = _numcell(f.values[j, idx...])
        end
    end
    vlabels = map(f.domains[1]) do x
        backend in (:latex, :markdown) ? latex(Event(v, x)) : string(v.name, "=", _valstr(x))
    end
    labels = Any[(_namecell(p, backend) for p in ps)..., vlabels...]
    title = sprint(show, Prob(Atom[v], Atom[ps...]))
    return (data=data, labels=labels, title=title,
            alignment=vcat(fill(:c, length(ps)), fill(:r, length(f.domains[1]))))
end

function _table(d::Derivation, backend)
    n = length(d)
    data = Matrix{Any}(undef, n, 3)
    for i in 1:n
        data[i, 1] = i - 1
        e = _mathcell(d.exprs[i], backend)
        data[i, 2] = i == 1 ? e : (e isa LaTeXString ? LaTeXString("\$= " * e[2:end]) : "= " * e)
        data[i, 3] = i == 1 ? "" : d.rules[i]
    end
    return (data=data, labels=Any["#", "expression", "justification"], title="",
            alignment=[:r, :l, :l])
end

function _table(xs::Independencies, backend)
    data = Matrix{Any}(undef, length(xs), 4)
    for (i, s) in enumerate(xs)
        data[i, 1] = join((v.name for v in s.X), ", ")
        data[i, 2] = backend === :latex ? LaTeXString("\$\\perp\\!\\!\\!\\perp\$") : "⟂"
        data[i, 3] = join((v.name for v in s.Y), ", ")
        data[i, 4] = isempty(s.Z) ? "∅" : join((v.name for v in s.Z), ", ")
    end
    return (data=data, labels=Any["", "", "", "given"], title="", alignment=[:r, :c, :l, :l])
end

"""
    tabulate([io,] x; backend = :text, kwargs...)

Render `x` as a table with PrettyTables. `backend` is `:text`, `:markdown`,
`:html` or `:latex`; other keyword arguments go to `pretty_table`. Without
`io`, returns the table as a `String`.

Works for [`Factor`](@ref) (one row per assignment), [`BayesNet`](@ref) (one
wide table per CPT), [`Derivation`](@ref), [`Independencies`](@ref) and
[`GenerativeModel`](@ref).
"""
tabulate(x; kwargs...) = sprint(io -> tabulate(io, x; kwargs...))

function tabulate(io::IO, x::Union{Factor,Derivation,Independencies}; backend::Symbol=:text, kwargs...)
    t = _table(x, backend)
    _pt(io, t, backend; kwargs...)
end

function tabulate(io::IO, bn::BayesNet; backend::Symbol=:text, kwargs...)
    for (k, v) in enumerate(topological_order(bn.dag))
        k > 1 && println(io)
        t = _cpt_table(bn.cpts[v], backend)
        _pt(io, t, backend; kwargs...)
    end
end

"""
    tabulate_cpt([io,] f::Factor; backend = :text)

Render a conditional probability table (first variable given the rest) in wide
form: one row per configuration of the parents.
"""
tabulate_cpt(f::Factor; kwargs...) = sprint(io -> tabulate_cpt(io, f; kwargs...))
tabulate_cpt(io::IO, f::Factor; backend::Symbol=:text, kwargs...) =
    _pt(io, _cpt_table(f, backend), backend; kwargs...)

function _pt(io::IO, t, backend; kwargs...)
    kw = (; column_labels=t.labels, alignment=t.alignment, backend=backend)
    # Expressions are long; wrapping in the terminal beats silently dropping
    # the justification column.
    backend === :text && (kw = merge(kw, (; fit_table_in_display_horizontally=false)))
    if !isempty(t.title)
        kw = merge(kw, (; title=(backend === :latex ? replace(t.title, "|" => "\$\\mid\$") : t.title)))
    end
    pretty_table(io, t.data; kw..., kwargs...)
end

# ---------------------------------------------------------------------------
# show methods
# ---------------------------------------------------------------------------

Base.show(io::IO, f::Factor) =
    print(io, "Factor(", join((v.name for v in f.vars), ", "), isempty(f.label) ? "" : "; " * f.label, ")")
function Base.show(io::IO, ::MIME"text/plain", f::Factor)
    isempty(f.vars) && return print(io, "Factor: ", f.values[])
    tabulate(io, f)
end
Base.show(io::IO, ::MIME"text/html", f::Factor) = tabulate(io, f; backend=:html)

Base.show(io::IO, bn::BayesNet) = print(io, "BayesNet(", length(bn.dag.nodes), " variables)")
function Base.show(io::IO, ::MIME"text/plain", bn::BayesNet)
    println(io, "BayesNet with ", length(bn.dag.nodes), " variables")
    tabulate(io, bn)
end
Base.show(io::IO, ::MIME"text/html", bn::BayesNet) = tabulate(io, bn; backend=:html)

function Base.show(io::IO, ::MIME"text/plain", d::Derivation)
    println(io, "Derivation with ", length(d) - 1, length(d) == 2 ? " step" : " steps")
    tabulate(io, d)
end
Base.show(io::IO, d::Derivation) = show(io, Equation(d))

Base.show(io::IO, ::MIME"text/plain", xs::Independencies) =
    (println(io, length(xs), " conditional independencies"); isempty(xs) || tabulate(io, xs))
function Base.show(io::IO, ::MIME"text/latex", xs::Independencies)
    print(io, "\$\$\\begin{aligned}", join((_latex(s) for s in xs), " \\\\ "), "\\end{aligned}\$\$")
end

# Tables.jl: one row per assignment, a column per variable plus `value`.
Tables.istable(::Type{Factor}) = true
Tables.columnaccess(::Type{Factor}) = true
function Tables.columns(f::Factor)
    rows = _assignments(f)
    cols = [Any[f.domains[k][idx[k]] for idx in rows] for k in eachindex(f.vars)]
    vals = Float64[f.values[idx...] for idx in rows]
    names = (Symbol[v.name for v in f.vars]..., :value)
    return NamedTuple{names}((cols..., vals))
end

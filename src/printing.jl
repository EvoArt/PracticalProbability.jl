# Unicode (REPL) and LaTeX rendering of atoms, expressions, equations and
# derivations. Tables (factors, CPTs, derivations in the REPL) are in tables.jl.

# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------

_valstr(v) = string(v)
_valstr(v::AbstractString) = repr(v)

# `inner`: the atom sits inside another compound event, so a compound needs brackets.
_astr(x::RV, inner=false) = string(x.name)
_astr(e::Event, inner=false) = inner ? string("{", e.rv.name, "=", _valstr(e.value), "}") :
                               string(e.rv.name, "=", _valstr(e.value))
_astr(i::Intervention, inner=false) = string("do(", _astr(i.target), ")")
_astr(::SureEvent, inner=false) = "Ω"
_astr(::ImpossibleEvent, inner=false) = "∅"
function _astr(c::Complement, inner=false)
    e = c.event
    e isa Union{EventUnion,EventIntersection,Complement} ? string("(", _astr(e), ")ᶜ") : string(_astr(e, true), "ᶜ")
end
function _astr(u::Union{EventUnion,EventIntersection}, inner=false)
    s = join((_astr(x, true) for x in u.parts), u isa EventUnion ? " ∪ " : " ∩ ")
    return inner ? string("(", s, ")") : s
end

# The letter: a named distribution keeps its name; the default is P for
# probabilities and p for densities of continuous variables.
# Does an atom stand for a continuous variable (so P(⋯) is a density p(⋯))?
_iscontinuous_atom(t::RV) = iscontinuous(t)
_iscontinuous_atom(::Atom) = false

function _letter(p::Prob)
    p.name === :P || return string(p.name)
    all(_iscontinuous_atom, p.targets) ? "p" : "P"
end

# Several events in P(⋅) are an intersection; write them with ∩ when they are
# named or compound events (`P(X=1, Y=2)` keeps the usual comma).
_joiner(atoms) = length(atoms) > 1 && all(iseventlike, atoms) && !all(a -> a isa Event, atoms) ? " ∩ " : ", "
_atomlist(atoms) = join((_astr(a, length(atoms) > 1 && _joiner(atoms) == " ∩ ") for a in atoms), _joiner(atoms))

Base.show(io::IO, x::Atom) = print(io, _astr(x))
function Base.show(io::IO, ::MIME"text/plain", x::RV)
    print(io, x.name, " (", x.kind === :event ? "event" : "$(x.kind) random variable", ")")
end

function Base.show(io::IO, p::Prob)
    print(io, _letter(p), "(", _atomlist(p.targets))
    isempty(p.given) || print(io, " | ", _atomlist(p.given))
    print(io, ")")
end

Base.show(io::IO, l::Literal) = print(io, l.value)

# Terms that print without brackets inside a product or quotient
# (expectation.jl adds E[⋯], Var(⋯), …).
_atomic(::Union{Prob,Literal}) = true
_atomic(::PExpr) = false

function _show_paren(io, e::PExpr, brackets=("[", "]"))
    _atomic(e) ? show(io, e) : (print(io, brackets[1]); show(io, e); print(io, brackets[2]))
end

function Base.show(io::IO, p::Product)
    fs = p.factors
    start = 1
    if fs[1] isa Literal && fs[1].value == -1
        print(io, "-")
        start = 2
    end
    for i in start:length(fs)
        f = fs[i]
        i > start && print(io, " ")
        _atomic(f) || (f isa Marginal && i == length(fs)) ? show(io, f) : _show_paren(io, f, ("(", ")"))
    end
end

# A term's sign and its absolute value, for printing sums.
function _signed(t::PExpr)
    c, r = _coef(t)
    (c isa Real && c < 0) || return (false, t)
    return (true, mul(Literal(-c), r))
end

function Base.show(io::IO, s::Sum)
    for (i, t) in enumerate(s.terms)
        neg, a = _signed(t)
        if i == 1
            neg && print(io, "-")
        else
            print(io, neg ? " - " : " + ")
        end
        a isa Sum ? _show_paren(io, a, ("(", ")")) : show(io, a)
    end
end

function Base.show(io::IO, q::Quotient)
    _show_paren(io, q.num)
    print(io, " / ")
    _show_paren(io, q.den)
end

_sub(vs) = length(vs) == 1 && length(string(vs[1].name)) == 1 ? string(vs[1].name) :
           string("{", join((v.name for v in vs), ","), "}")

function Base.show(io::IO, m::Marginal)
    d = filter(isdiscrete, m.over)
    c = filter(iscontinuous, m.over)
    isempty(d) || print(io, "∑_", _sub(d), " ")
    isempty(c) || print(io, "∫ ")
    m.body isa Union{Quotient,Sum} ? _show_paren(io, m.body) : show(io, m.body)
    for v in c
        print(io, " d", v.name)
    end
end

function Base.show(io::IO, e::Equation)
    show(io, e.lhs)
    print(io, e.relation === :propto ? " ∝ " : " = ")
    show(io, e.rhs)
end

function Base.show(io::IO, s::Independence)
    print(io, join((v.name for v in s.X), ", "), " ⟂ ", join((v.name for v in s.Y), ", "))
    isempty(s.Z) || print(io, " | ", join((v.name for v in s.Z), ", "))
end

function Base.show(io::IO, g::Given)
    print(io, _atomlist(g.targets), " | ", _atomlist(g.given))
end

function Base.show(io::IO, g::DAG)
    print(io, "DAG(", length(g.nodes), " nodes, ", length(edges(g)), " edges)")
end
function Base.show(io::IO, ::MIME"text/plain", g::DAG)
    println(io, "DAG with ", length(g.nodes), " nodes and ", length(edges(g)), " edges:")
    order = topological_order(g)
    for v in order
        ps = g.parents[v]
        print(io, "  ", v.name, isempty(ps) ? "" : " ← " * join((p.name for p in ps), ", "))
        v === last(order) || println(io)
    end
end

# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

const _LATEXIFY_KW = (env=:raw, index=:subscript, cdot=false)

_lx(x) = String(latexify(x; _LATEXIFY_KW...))
_lx(x::AbstractString) = string("\\text{", x, "}")

_lxatom(x::RV, inner=false) = _lx(x.name)
_lxatom(e::Event, inner=false) = inner ? string("\\{", _lx(e.rv.name), " = ", _lx(e.value), "\\}") :
                                 string(_lx(e.rv.name), " = ", _lx(e.value))
_lxatom(i::Intervention, inner=false) = string("\\mathrm{do}(", _lxatom(i.target), ")")
_lxatom(::SureEvent, inner=false) = "\\Omega"
_lxatom(::ImpossibleEvent, inner=false) = "\\varnothing"
function _lxatom(c::Complement, inner=false)
    e = c.event
    base = e isa Union{EventUnion,EventIntersection,Complement} ? string("(", _lxatom(e), ")") : _lxatom(e, true)
    return string(base, "^{\\mathsf{c}}")
end
function _lxatom(u::Union{EventUnion,EventIntersection}, inner=false)
    s = join((_lxatom(x, true) for x in u.parts), u isa EventUnion ? " \\cup " : " \\cap ")
    return inner ? string("(", s, ")") : s
end

_lxjoiner(atoms) = _joiner(atoms) == " ∩ " ? " \\cap " : ", "
_lxatomlist(atoms) = join((_lxatom(a, _joiner(atoms) == " ∩ ") for a in atoms), _lxjoiner(atoms))

_lxletter(p::Prob) = p.name === :P ? _letter(p) : _lx(p.name)

function _latex(p::Prob)
    s = string(_lxletter(p), "(", _lxatomlist(p.targets))
    isempty(p.given) || (s *= string(" \\mid ", _lxatomlist(p.given)))
    return s * ")"
end
_latex(l::Literal) = _lx(l.value)
_lxparen(e::PExpr) = _atomic(e) ? _latex(e) : string("\\left(", _latex(e), "\\right)")
function _latex(p::Product)
    fs = p.factors
    parts = String[]
    start = 1
    neg = fs[1] isa Literal && fs[1].value == -1
    neg && (start = 2)
    for i in start:length(fs)
        f = fs[i]
        push!(parts, _atomic(f) || f isa Quotient ||
                     (f isa Marginal && i == length(fs)) ? _latex(f) : _lxparen(f))
    end
    return (neg ? "-" : "") * join(parts, " \\, ")
end
function _latex(s::Sum)
    out = ""
    for (i, t) in enumerate(s.terms)
        neg, a = _signed(t)
        body = a isa Sum ? _lxparen(a) : _latex(a)
        out *= i == 1 ? (neg ? "-" : "") * body : (neg ? " - " : " + ") * body
    end
    return out
end
_latex(q::Quotient) = string("\\frac{", _latex(q.num), "}{", _latex(q.den), "}")
function _latex(m::Marginal)
    d = filter(isdiscrete, m.over)
    c = filter(iscontinuous, m.over)
    s = ""
    isempty(d) || (s *= string("\\sum_{", join((_lx(v.name) for v in d), ", "), "} "))
    for _ in c
        s *= "\\int "
    end
    s *= m.body isa Sum ? _lxparen(m.body) : _latex(m.body)
    for v in c
        s *= string(" \\, \\mathrm{d}", _lx(v.name))
    end
    return s
end
_latex(e::Equation) = string(_latex(e.lhs), e.relation === :propto ? " \\propto " : " = ", _latex(e.rhs))
function _latex(s::Independence)
    l = string(join((_lx(v.name) for v in s.X), ", "), " \\perp\\!\\!\\!\\perp ",
               join((_lx(v.name) for v in s.Y), ", "))
    isempty(s.Z) ? l : string(l, " \\mid ", join((_lx(v.name) for v in s.Z), ", "))
end
_latex(a::Atom) = _lxatom(a)

# A justification in text mode. Variable names may carry superscripts
# (`x₁^*`), which text mode rejects ("'^' allowed only in math mode" in
# MathJax, an error in LaTeX); they go into inline math, `x₁$^{*}$`.
function _lxrule(r)
    s = replace(r, "_" => "\\_", "&" => "\\&", "%" => "\\%", "#" => "\\#")
    s = replace(s, r"\^(\{[^{}]*\}|[^\s{])" => m -> string("\$^{", strip(m[2:end], ['{', '}']), "}\$"))
    return string("\\text{", s, "}")
end

function _latex(d::Derivation)
    lines = String[]
    push!(lines, string(_latex(first(d)), " &= ", length(d) > 1 ? _latex(d.exprs[2]) : _latex(first(d)),
                        length(d) > 1 ? string(" && ", _lxrule(d.rules[2])) : ""))
    for i in 3:length(d)
        push!(lines, string("&= ", _latex(d.exprs[i]), " && ", _lxrule(d.rules[i])))
    end
    return string("\\begin{aligned}\n", join(lines, " \\\\\n"), "\n\\end{aligned}")
end

function _wrap(s::AbstractString, env::Symbol)
    env === :raw && return LaTeXString(s)
    env === :inline && return LaTeXString(string("\$", s, "\$"))
    env === :display && return LaTeXString(string("\$\$", s, "\$\$"))
    env === :equation && return LaTeXString(string("\\begin{equation}\n", s, "\n\\end{equation}"))
    throw(ArgumentError("env must be :raw, :inline, :display or :equation"))
end

"""
    latex(x; env = :inline) -> LaTeXString

LaTeX for a probability expression, event, [`Equation`](@ref),
[`Derivation`](@ref), [`Independence`](@ref) statement,
[`GenerativeModel`](@ref) or table. `env`: `:inline` (`\$…\$`), `:display`
(`\$\$…\$\$`), `:equation` or `:raw` (no delimiters).

`latexify` from Latexify.jl also works on all of these.
"""
latex(x; env::Symbol=:inline) = _wrap(_latex(x), env)

const _LatexShowable = Union{PExpr,Equation,Derivation,Independence,Atom}
Base.show(io::IO, ::MIME"text/latex", x::_LatexShowable) = print(io, "\$\$", _latex(x), "\$\$")

Latexify.latexify(x::_LatexShowable; env::Symbol=:inline, kwargs...) = latex(x; env)

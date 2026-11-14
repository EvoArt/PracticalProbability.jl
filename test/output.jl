@testset "output" begin
    @rv A B C θ::continuous σ::continuous

    @testset "LaTeX" begin
        @test latex(P(A, B | C)) == raw"$P(A, B \mid C)$"
        @test latex(P(A | B); env=:raw) == raw"P(A \mid B)"
        @test latex(P(θ | A); env=:raw) == raw"p(\theta \mid A)"
        @test latex(P(A(1) | Do(B)); env=:raw) == raw"P(A = 1 \mid \mathrm{do}(B))"
        @test latex(P(A | B) * P(B) / P(A); env=:raw) == raw"\frac{P(A \mid B) \, P(B)}{P(A)}"
        @test latex(Σ(B, P(A, B)); env=:raw) == raw"\sum_{B} P(A, B)"
        @test latex(Σ([θ, σ], P(θ, σ)); env=:raw) == raw"\int \int p(\theta, \sigma) \, \mathrm{d}\theta \, \mathrm{d}\sigma"
        @test latex(A ⟂ B | C; env=:raw) == raw"A \perp\!\!\!\perp B \mid C"
        @test startswith(latex(P(A); env=:display), "\$\$")
        @test latexify(P(A | B)) == latex(P(A | B))

        d = simplify(bayes(P(A | B)); trace=true)
        s = latex(d; env=:raw)
        @test startswith(s, "\\begin{aligned}") && occursin("&&", s) && occursin("\\text{chain rule", s)
        @test sprint(show, MIME"text/latex"(), P(A)) == "\$\$P(A)\$\$"

        # superscripts in names go into inline math inside \text (MathJax and
        # LaTeX reject `^` in text mode)
        lx = PracticalProbability._lxrule
        @test lx("marginalise x₁^* out") == "\\text{marginalise x₁\$^{*}\$ out}"
        @test lx("a^{ab} b_c") == "\\text{a\$^{ab}\$ b\\_c}"
        xs = RV(Symbol("x^*"))
        ds = latex(simplify(Σ(xs, P(xs) * P(A | xs)); trace=true); env=:raw)
        @test !occursin(r"\\text\{[^$}]*\^", ds)
    end

    @testset "text" begin
        @test sprint(show, bayes(P(A | B))) == "[P(B | A) P(A)] / P(B)"
        @test sprint(show, Σ(B, P(A | B) * P(B))) == "∑_B P(A | B) P(B)"
        @test sprint(show, Σ(θ, P(A | θ) * P(θ))) == "∫ P(A | θ) p(θ) dθ"
        @test sprint(show, A ⟂ B | C) == "A ⟂ B | C"
        @test sprint(show, Equation(P(A), Σ(B, P(A, B)))) == "P(A) = ∑_B P(A, B)"
    end

    @testset "tables" begin
        bn = BayesNet(A => [0.25, 0.75], (B | A) => [0.5 0.1; 0.5 0.9])
        f = compute(P(B | A), bn)
        t = tabulate(f)
        @test occursin("P(B | A)", t) && occursin("0.9", t)
        @test occursin("<table", tabulate(f; backend=:html))
        @test occursin("\\begin{tabular}", tabulate(f; backend=:latex))
        md = tabulate(f; backend=:markdown)
        @test occursin("\\mid", md)      # math, not a bare `|` that would split the cell
        @test occursin("P(B | A)", tabulate(bn)) && occursin("B=2", tabulate(bn))
        @test occursin("B=1", tabulate_cpt(cpt(bn, B)))

        d = simplify(bayes(P(A | B)); trace=true)
        td = tabulate(d)
        @test occursin("justification", td) && occursin("= P(A | B)", td)
        @test occursin("\\begin{tabular}", tabulate(d; backend=:latex))

        g = @dag A → B → C
        ti = sprint(show, MIME"text/plain"(), independencies(g))
        @test occursin("⟂", ti)

        # rich display hooks exist for notebooks
        @test showable(MIME"text/html"(), f) && showable(MIME"text/latex"(), d)
        @test occursin("<table", sprint(show, MIME"text/html"(), bn))
    end
end

# Drawing the diagrams with PracticalBayesPlot (the default backend): every
# kind renders to well-formed SVG and to TikZ, and the Venn cells are exact.

const PBP = PracticalProbability.PBP

function wellformed_svg(s::AbstractString)
    stack = String[]
    for m in eachmatch(r"<(/?)([A-Za-z]+)[^>]*?(/?)>", s)
        closing, name, selfclosing = m.captures
        if closing == "/"
            (isempty(stack) || pop!(stack) != name) && return false
        elseif selfclosing != "/"
            push!(stack, name)
        end
    end
    return isempty(stack)
end

balanced_tikz(t) = count(==('{'), t) == count(==('}'), t) && count(==('['), t) == count(==(']'), t) &&
                   startswith(t, "\\begin{tikzpicture}") && !occursin("NaN", t)

# area of a polygon (shoelace)
polyarea(P) = abs(sum(P[i][1] * P[mod1(i + 1, end)][2] - P[mod1(i + 1, end)][1] * P[i][2] for i in eachindex(P))) / 2

@testset "drawing (PracticalBayesPlot)" begin
    @test plotbackend() in (:pbp, :makie)
    @test_throws ArgumentError plotbackend!(:gr)
    if Base.get_extension(PracticalProbability, :PracticalProbabilityMakieExt) === nothing
        @test plotbackend() === :pbp                       # the default
        @test_throws ArgumentError plotbackend!(:makie)    # no Makie loaded
    end
    plotbackend!(:pbp)
    @test plotbackend() === :pbp

    @event A B C
    @rv X Y Z
    g = DAG(Z => X, Z => Y, X => Y)
    test = BayesNet(A => 0.01, (B | A) => [0.95, 0.05])
    bn = BayesNet(X => [0.3, 0.7], (Y | X) => [0.9 0.2; 0.1 0.8])

    figs = [
        "dag" => dagplot(g),
        "backdoor" => dagplot(g, P(Y | Do(X))),
        "dsep" => dagplot(DAG(X => Y, Y => Z), X ⟂ Z | Y),
        "tree" => probtree(P(A | B); bn=test),
        "tree-levels" => probtree([A, B]; labels=:text),
        "mosaic" => mosaic(P(A | B); bn=test),
        "venn" => venn(P(A | B)),
        "venn3" => venn(P(A ∩ B | C)),
        "bayes" => venn(Equation(P(A | B), bayes(P(A | B)))),
        "impossible" => venn(P(X(1) ∪ X(2))),
        "bars" => factorplot(compute(P(Y | X), bn)),
        "heatmap" => factorplot(compute(P(Y | X), bn); kind=:heatmap),
        "one" => factorplot(compute(P(X), bn)),
        "three" => factorplot(compute(P(X, Y, Z), rand_bayesnet(g))),
    ]
    for (name, fig) in figs
        @test fig isa PBP.Figure
        s = PBP.svg(fig)
        @test wellformed_svg(s)
        @test !occursin("NaN", s)
        t = PBP.tikz(fig)
        @test balanced_tikz(t)
    end

    # LaTeX labels go to TikZ, plain ones to SVG
    s = PBP.svg(venn(P(A | B)))
    t = PBP.tikz(venn(P(A | B)))
    @test occursin(">P(A | B)<", s) || occursin(">P(A ∣ B)<", s)
    @test occursin("\\mid", t)

    # into an existing axis
    fig = PBP.Figure(; size=(600, 300))
    ax = PBP.Axis(fig[1, 1]; aspect=PBP.DataAspect())
    V = dagplot!(ax, g)
    @test V isa DAGView
    @test !isempty(ax.plots)
    L = probtree!(PBP.Axis(fig[1, 2]), [A, B])
    @test L isa TreeLayout
    @test vennplot!(PBP.Axis(fig[2, 1]), P(A)) isa VennLayout
    @test mosaic!(PBP.Axis(fig[2, 2]), P(A | B); bn=test) isa MosaicLayout
    @test wellformed_svg(PBP.svg(fig))
    @test_throws ArgumentError factorplot!(PBP.Axis(fig[3, 1]), compute(P(Y | X), bn); kind=:pie)

    # Venn geometry: the lens of two unit-panel circles has the analytic area
    PP = PracticalProbability
    c1 = PP._pcircle(-0.15, 0.0, 0.25; n=720)
    c2 = PP._pcircle(0.15, 0.0, 0.25; n=720)
    lens = PP._intersect_convex(c1, c2)
    d, r = 0.3, 0.25
    exact = 2r^2 * acos(d / 2r) - d / 2 * sqrt(4r^2 - d^2)
    @test polyarea(lens) ≈ exact rtol = 1e-3
    @test isempty(PP._intersect_convex(PP._pcircle(0.0, 0.0, 0.1), PP._pcircle(1.0, 0.0, 0.1)))
    # a half-plane cut of the unit square
    sq = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    @test polyarea(PP._keep(sq, p -> 0.25 - p[1])) ≈ 0.25

    # colours composite onto white: the given cell of P(A | B) is a light tint
    st = PP._pvennstyle()
    @test all(>(0.8), st.given) && st.highlight == PP._rgb3(1)
    @test st.excluded == PP._over((140 / 255, 140 / 255, 140 / 255), 0.4, (1.0, 1.0, 1.0))
end

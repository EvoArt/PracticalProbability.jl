# Venn diagrams: the layout (pure Julia) always; the Makie extension when
# CairoMakie can be loaded (it is not a test dependency, so these run only in
# an environment that has it, e.g. `julia --project=<env with CairoMakie> test/venn.jl`).

using PracticalProbability: venn_layout, venn_sets, VennLayout, VennPanel, VENN_MAXSETS,
                            _venn_cell, _venn_geometry, _ncells, _allcells, VENN_BOX

# bitmask of a list of cells
cells(cs...) = reduce(|, (UInt32(1) << c for c in cs); init=UInt32(0))

@testset "Venn diagrams" begin
    @rv A::event B::event C::event D::event X Y

    @testset "sets" begin
        @test venn_sets(P(A | B)) == [A, B]
        @test venn_sets(P(B ∪ A' | C)) == [B, A, C]
        @test venn_sets(Equation(P(A, B), P(A | B) * P(B))) == [A, B]
        @test venn_sets(P(X(1) | (X(2) ∪ A))) == [X(1), X(2), A]
        @test isempty(venn_sets(Literal(1)))
        @test_throws ArgumentError venn_sets(P(X))                 # a variable, not an event
        @test_throws ArgumentError venn_sets(P(A | Do(X(1))))
        @test_throws ArgumentError venn_sets(Σ(X, P(A, X)))
        # an atom type venn knows nothing about (and that cannot even print) is an ArgumentError
        @eval struct _UnknownAtom <: PracticalProbability.Atom end
        @test_throws ArgumentError venn_sets(P(A | _UnknownAtom()))
        @test_throws ArgumentError venn_layout(P(A, B, C, D))       # too many circles
        @test_throws ArgumentError venn_layout(P(A); sets=[B])      # sets must cover the expression
        @test venn_layout(P(A); sets=[B, A]).sets == [B, A]
    end

    @testset "geometry: every cell is drawn" for n in 0:VENN_MAXSETS
        hw, hh = VENN_BOX
        seen = Set{Int}()
        for x in range(-hw, hw; length=201), y in range(-hh, hh; length=145)
            push!(seen, _venn_cell(n, x, y))
        end
        @test seen == Set(0:_ncells(n)-1)
        centres, r, labels = _venn_geometry(n)
        for (cx, cy) in centres                 # circles stay inside the box
            @test abs(cx) + r < hw && abs(cy) + r < hh
        end
        for (lx, ly) in labels                  # names sit inside the box, outside every circle
            @test abs(lx) < hw && abs(ly) < hh
            @test _venn_cell(n, lx, ly) == 0
        end
    end

    @testset "regions" begin
        # cell bits: A = 1, B = 2
        L = venn_layout(P(A ∪ B))
        @test only(L.panels).target == cells(1, 2, 3)
        @test !only(L.panels).conditional
        @test only(L.panels).given == _allcells(2)

        L = venn_layout(P(A | B))
        p = only(L.panels)
        @test p.conditional
        @test p.given == cells(2, 3)
        @test p.target == cells(3)

        L = venn_layout(P(A' | B))
        @test only(L.panels).target == cells(2)
        L = venn_layout(P((A ∪ B)'))
        @test only(L.panels).target == cells(0)
        L = venn_layout(P(A ∩ B'))
        @test only(L.panels).target == cells(1)

        # three circles: A = 1, B = 2, C = 4
        L = venn_layout(P(A ∩ B | C))
        @test only(L.panels).target == cells(7)
        @test only(L.panels).given == cells(4, 5, 6, 7)

        # X = 1 and X = 2 cannot both happen: their overlap is impossible
        L = venn_layout(P(X(1) ∪ X(2)))
        @test L.feasible == cells(0, 1, 2)
        @test only(L.panels).target == cells(1, 2)
        @test venn_layout(P(X(1) ∪ Y(1))).feasible == _allcells(2)
    end

    @testset "layouts" begin
        ie = PracticalProbability.inclusion_exclusion(P(A ∪ B))
        L = venn_layout(Equation(P(A ∪ B), ie))
        @test length(L.panels) == 4
        @test [t.text for t in L.texts if t.role === :op] == ["=", "+", "−"]
        @test L.panels[1].target == cells(1, 2, 3)
        @test Set(p.target for p in L.panels[2:end]) == Set([cells(1, 3), cells(2, 3), cells(3)])
        # panels left to right, all on one row
        @test issorted([p.x for p in L.panels])
        @test all(p -> p.y == 0, L.panels)

        # Bayes' rule: a fraction, numerator above the bar, denominator below
        L = venn_layout(Equation(P(A | B), bayes(P(A | B))))
        @test length(L.panels) == 4
        @test length(L.lines) == 1
        bar = only(L.lines)
        @test count(p -> p.y > bar.y0, L.panels) == 2
        @test count(p -> p.y < bar.y0, L.panels) == 1
        xmin, xmax, ymin, ymax = L.bbox
        @test all(p -> xmin <= p.x - 0.5 && p.x + 0.5 <= xmax && ymin <= p.y - 0.5 && p.y + 0.36 <= ymax, L.panels)

        # sums inside products get brackets
        L = venn_layout(P(C) * (P(A) + P(B)))
        @test length(L.lines) == 6

        # coefficients and literals
        L = venn_layout(1 - P(A'))
        @test [t.text for t in L.texts] == ["1", "−"]
        L = venn_layout(2 * P(A))
        @test [t.text for t in L.texts] == ["2"]

        # a derivation: one row per step, justifications as notes
        d = derivation(P(A ∪ B))
        apply!(d, PracticalProbability.inclusion_exclusion, "inclusion–exclusion")
        L = venn_layout(d)
        @test length(L.panels) == 4
        @test [t.text for t in L.texts if t.role === :note] == ["inclusion–exclusion"]
        @test length(unique(p.y for p in L.panels)) == 2

        @test venn_layout(L) === L
        @test sprint(show, L) == "VennLayout(2 sets: A, B; 4 panels)"
    end

    if Base.find_package("CairoMakie") !== nothing
        @eval using CairoMakie: CairoMakie
        ext = Base.get_extension(PracticalProbability, :PracticalProbabilityMakieExt)
        @testset "Makie extension" begin
            @test ext !== nothing
            # drawing.jl chose :pbp explicitly, so loading Makie did not switch
            @test plotbackend() === :pbp
            plotbackend!(:makie)
            @test plotbackend() === :makie

            # labels are LaTeX that MathTeXEngine can lay out (not the plain-text fallback)
            for x in (P(A | B), P(A ∩ B' | C), P((A ∪ B)'), P(X(1) ∪ X(2)), A, X(1))
                @test ext._label(x, :latex) isa PracticalProbability.LaTeXString
            end
            @test ext._label(P(A | B), :text) == "P(A | B)"

            # the fill is right where it matters: probe the middle of each cell
            L = venn_layout(P(A | B))
            p = only(L.panels)
            colours = (highlight=CairoMakie.RGBAf(1, 0, 0, 1), given=CairoMakie.RGBAf(0, 1, 0, 1),
                       excluded=CairoMakie.RGBAf(0, 0, 1, 1), impossible=CairoMakie.RGBAf(0, 0, 0, 1))
            img, (x0, x1), (y0, y1) = ext._panel_image(2, p, L.feasible, colours; pixels=200)
            at(x, y) = img[round(Int, (x - x0) / (x1 - x0) * size(img, 1)), round(Int, (y - y0) / (y1 - y0) * size(img, 2))]
            @test at(0.0, 0.0) == colours.highlight          # A ∩ B
            @test at(0.3, 0.0) == colours.given              # B only
            @test at(-0.3, 0.0) == colours.excluded          # A only: outside the condition
            @test at(0.45, 0.32) == colours.excluded         # neither
            # the fill is opaque (composited onto the background), with blended edges
            @test all(c -> CairoMakie.alpha(c) == 1, img)
            @test length(unique(img)) > 4
            # translucent colours are composited onto the background, not left translucent
            soft = merge(colours, (excluded=CairoMakie.RGBAf(0, 0, 0, 0.5),))
            img, _, _ = ext._panel_image(2, p, L.feasible, soft; pixels=200,
                                         background=CairoMakie.RGBAf(1, 1, 1, 1))
            c = at(0.45, 0.32)
            @test all(isapprox(f(c), 0.5; atol=1e-3) for f in (CairoMakie.red, CairoMakie.green, CairoMakie.blue))
            @test CairoMakie.alpha(c) == 1

            L = venn_layout(P(X(1) ∪ X(2)))
            img, _, _ = ext._panel_image(2, only(L.panels), L.feasible, colours; pixels=200)
            @test count(==(colours.impossible), img) > 0     # the hatched impossible overlap

            # whole figures render, for every kind of input
            dir = mktempdir()
            d = derivation(P(A ∪ B))
            apply!(d, PracticalProbability.inclusion_exclusion, "inclusion–exclusion")
            inputs = Any[P(A), P(A | B), P(A ∩ B' | C), P(X(1) ∪ X(2)), 1 - P(A'),
                         Equation(P(A ∪ B), PracticalProbability.inclusion_exclusion(P(A ∪ B))),
                         Equation(P(A | B), bayes(P(A | B))), P(C) * (P(A) + P(B)), d]
            for (i, x) in enumerate(inputs)
                fap = venn(x)
                @test fap isa CairoMakie.Makie.FigureAxisPlot
                f = joinpath(dir, "venn$i.png")
                CairoMakie.save(f, fap)
                @test filesize(f) > 1000
            end
            for labels in (:text, :none)
                @test venn(P(A | B); labels) isa CairoMakie.Makie.FigureAxisPlot
            end
            @test_throws ArgumentError venn(P(A); labels=:nope)

            # the recipe on its own, in a user's axis
            fig = CairoMakie.Figure()
            ax = CairoMakie.Axis(fig[1, 1]; aspect=CairoMakie.DataAspect())
            @test vennplot!(ax, P(A | B); highlightcolor=:orange) isa CairoMakie.Makie.AbstractPlot
            CairoMakie.save(joinpath(dir, "recipe.png"), fig)
            @test_throws ArgumentError venn(P(X))
        end
    else
        @info "CairoMakie not in the test environment; skipping the Makie extension tests"
    end
end

# DAG views, probability trees, mosaics, sample events and Venn numbers: the
# layouts always (pure Julia); drawing only where CairoMakie loads.

const PPd = PracticalProbability

@testset "diagrams" begin
    @testset "d-separation paths agree with d_separated" begin
        rng = Xoshiro(7)
        for trial in 1:60
            n = rand(rng, 4:7)
            vs = [RV(Symbol("V", k)) for k in 1:n]
            g = DAG([vs[i] => vs[j] for i in 1:n for j in i+1:n if rand(rng) < 0.4]...; nodes=vs)
            perm = vs[sortperm(rand(rng, n))]
            X, Y = perm[1:1], perm[2:2]
            Z = perm[3:2+rand(rng, 0:2)]
            paths = PPd.dag_paths(g, X, Y, Z)
            @test all(p -> !p.open, paths) == d_separated(g, X, Y, Z)
            for p in paths
                @test first(p.nodes) in X && last(p.nodes) in Y && allunique(p.nodes)
                @test p.open == (p.blocker === nothing)
            end
        end
    end

    @testset "paths: chains, forks, colliders" begin
        @rv A B C D
        chain = @dag A → B → C
        p = only(PPd.dag_paths(chain, A, C, [B]))
        @test !p.open && p.blocker == B && occursin("chain", p.reason)
        @test only(PPd.dag_paths(chain, A, C)).open
        @test PPd.pathstring(chain, p) == "A → B → C"
        fork = @dag B → (A, C)
        @test occursin("fork", only(PPd.dag_paths(fork, A, C, [B])).reason)
        coll = @dag begin
            (A, C) → B
            B → D
        end
        @test !only(PPd.dag_paths(coll, A, C)).open                 # a collider blocks
        @test only(PPd.dag_paths(coll, A, C, [B])).open             # unless conditioned on
        @test only(PPd.dag_paths(coll, A, C, [D])).open             # or a descendant is
        @test PPd.pathstring(coll, only(PPd.dag_paths(coll, A, C))) == "A → B ← C"
    end

    @testset "dag_view" begin
        @rv A B C X Y Z
        g = @dag A → B → C
        V = PPd.dag_view(g, A ⟂ C | B)
        @test length(V.panels) == 1 && length(V.paths) == 1
        panel = only(V.panels)
        nb = only(n for n in panel.nodes if n.v == B)
        @test nb.fill === :observed && nb.blocks
        @test all(e -> e.style === :blocked, panel.edges)
        @test count(n -> n.endpoint, panel.nodes) == 2
        @test any(t -> t.role === :caption && occursin("⟂", t.text), V.texts)
        @test any(t -> occursin("holds", t.text), V.texts)
        V2 = PPd.dag_view(g, A ⟂ C)
        @test all(e -> e.style === :open, only(V2.panels).edges)
        @test any(t -> occursin("does not hold", t.text), V2.texts)

        # layers: parents above children; no two nodes in the same place
        big = @dag begin
            A → (B, C)
            B → (X, Y)
            C → Y
            X → Z
        end
        P0 = only(PPd.dag_view(big).panels)
        pos = Dict(n.v => (n.x, n.y) for n in P0.nodes)
        @test allunique(values(pos))
        @test all(pos[p][2] > pos[c][2] for (p, c) in PPd.edges(big))

        # back-door adjustment and graph surgery
        conf = @dag begin
            Z → (X, Y)
            X → Y
        end
        V = PPd.dag_view(conf, P(Y | Do(X)))
        @test length(V.panels) == 2
        bd = only(V.paths)
        @test bd.nodes == [X, Z, Y] && !bd.open && bd.blocker == Z
        left, right = V.panels
        @test only(n for n in left.nodes if n.v == Z).fill === :highlight
        @test only(e for e in right.edges if e.from == Z && e.to == X).style === :removed
        @test only(n for n in right.nodes if n.v == X).shape === :square
        @test minimum(n.x for n in right.nodes) > maximum(n.x for n in left.nodes)
        # each title sits over its own panel
        for (panel, t) in zip(V.panels, filter(t -> t.role === :title, V.texts))
            b = PPd._panelbox(panel)
            @test b[1] <= t.x <= b[2]
        end
        # Z → Y would run straight through X (they are stacked): it bends; no straight edge crosses a node
        @test only(e for e in left.edges if e.from == Z && e.to == Y).bend !== nothing
        for panel in V.panels
            pos = Dict(n.v => (n.x, n.y) for n in panel.nodes)
            for e in panel.edges
                e.bend === nothing || continue
                @test all(PPd._segdist(pos[v], pos[e.from], pos[e.to]) >= PPd.DAG_RADIUS
                          for v in keys(pos) if v != e.from && v != e.to)
            end
        end
        @test any(t -> t.role === :caption && t.latex !== nothing && occursin("sum", t.latex), V.texts)
        @test_throws ArgumentError PPd.dag_view(conf, P(Y | X))

        # plates on a plain DAG: the box holds its members and nothing else
        pl = @dag begin
            A → X
            X → Y
            B → Y
        end
        V = PPd.dag_view(pl; plates=Dict(:i => [X, Y]))
        P1 = only(V.panels)
        box = only(P1.plates)
        inside(n) = box.x0 < n.x < box.x1 && box.y0 < n.y < box.y1
        @test all(n -> inside(n) == (n.v in (X, Y)), P1.nodes)
        @test box.label == "i"
    end

    begin
        @testset "dag_view of a model with plates" begin
            hier = @generative function hier(J, y, sigma)
                mu ~ Normal(0, 5)
                tau ~ truncated(Cauchy(0, 5); lower=0)
                for j in 1:J
                    theta[j] ~ Normal(mu, tau)
                    y[j] ~ Normal(theta[j], sigma[j])
                end
            end
            V = PPd.dag_view(hier)
            P1 = only(V.panels)
            box = only(P1.plates)
            @test box.label == "j = 1:J"
            byname = Dict(n.v.name => n for n in P1.nodes)
            inside(n) = box.x0 < n.x < box.x1 && box.y0 < n.y < box.y1
            @test inside(byname[:theta]) && inside(byname[:y]) && !inside(byname[:mu]) && !inside(byname[:tau])
            @test byname[:y].fill === :observed && byname[:theta].fill === :none
        end
    end

    @testset "probability trees" begin
        @event D T
        test = BayesNet(D => 0.01, (T | D) => [0.95, 0.05])
        L = PPd.tree_layout([D, T]; bn=test)
        @test length(L.leaves) == 4 && length(L.branches) == 6
        @test sum(l.joint for l in L.leaves) ≈ 1
        @test only(l.joint for l in L.leaves if l.path == [D, T]) ≈ 0.01 * 0.95
        @test all(l -> l.role === :none, L.leaves)
        # leaves top to bottom, children at their branch ends
        @test issorted([l.y for l in L.leaves]; rev=true)
        @test all(b -> b.x1 > b.x0 && isfinite(b.y0), L.branches)

        L = PPd.tree_layout(P(D | T); bn=test)
        @test L.levels == [D, T]
        @test L.result ≈ compute(P(D | T), test)
        @test [l.role for l in L.leaves] == [:target, :excluded, :given, :excluded]
        @test occursin("=", L.caption) && L.captionlatex !== nothing && occursin("frac", L.captionlatex)

        # numbers from facts: what deduce knows, and `nothing` where it does not
        @event A B
        facts = (P(A) => 0.3, P(B) => 0.5, P(A ∪ B) => 0.6)
        L = PPd.tree_layout(P(A | B); order=[B, A], facts)
        @test L.result ≈ Float64(deduce(P(A | B), facts...))
        L = PPd.tree_layout([A, B]; facts=(P(A) => 0.3,))
        @test all(isnothing, (b.value for b in L.branches if b.atom in (B, B')))
        @test L.result === nothing

        @rv X
        @test_throws ArgumentError PPd.tree_layout([X])                      # values need a BayesNet
        @test_throws ArgumentError PPd.tree_layout([A]; bn=test, facts)      # one source of numbers

        # labels: none overlaps another or crosses a branch (three outcomes give a middle branch)
        @rv W R
        three = BayesNet(W => [0.2, 0.5, 0.3], (R | W) => [0.9 0.5 0.1; 0.1 0.5 0.9];
                         domains=Dict(W => [:low, :mid, :high], R => [:dry, :wet]))
        F = PPd.TREE_FONT
        overlaps(a, b) = abs(a[1] - b[1]) < (a[3] + b[3]) / 2 && abs(a[2] - b[2]) < (a[4] + b[4]) / 2
        # does the segment cross the box (centre x, y, width, height)? sampled finely
        crosses(b, box) = any(range(0, 1; length=400)) do t
            x, y = b.x0 + t * (b.x1 - b.x0), b.y0 + t * (b.y1 - b.y0)
            abs(x - box[1]) < box[3] / 2 && abs(y - box[2]) < box[4] / 2
        end
        for L in (PPd.tree_layout(P(D | T); bn=test), PPd.tree_layout([W, R]; bn=three))
            mid = PPd._branchmiddle(L)
            @test count(mid) == (L.levels[1] == W ? 1 : 0)     # W = mid, between low and high
            boxes = NTuple{4,Float64}[]
            for (b, m) in zip(L.branches, mid)
                wp, wo = PPd._textwidth(PPd._branchtext(b), F.prob), PPd._textwidth(sprint(show, b.atom), F.outcome)
                (px, py), (ox, oy) = PPd._branchlabels(b, wp, F.prob, wo, F.outcome; middle=m)
                append!(boxes, [(px, py, wp, F.prob), (ox, oy, wo, F.outcome)])
            end
            for l in L.leaves
                x0, w = PPd._leafbox(l)
                push!(boxes, (x0 + w / 2, l.y, w, 0.44))
            end
            @test !any(overlaps(boxes[i], boxes[j]) for i in eachindex(boxes) for j in i+1:length(boxes))
            @test !any(crosses(b, box) for b in L.branches for box in boxes)
            @test L.bbox[2] >= maximum(box[1] + box[3] / 2 for box in boxes)
        end
    end

    @testset "mosaics" begin
        @event D T
        test = BayesNet(D => 0.01, (T | D) => [0.95, 0.05])
        M = PPd.mosaic_layout(P(D | T); bn=test)
        area(r) = (r[2] - r[1] + 2PPd.MOSAIC_GAP) * (r[4] - r[3] + 2PPd.MOSAIC_GAP)
        @test all(area(r) ≈ t.joint for (t, r) in zip(M.tiles, M.rects))
        @test sum(area, M.rects) ≈ 1
        @test M.result ≈ compute(P(D | T), test)
        @test [h[1] for h in M.headers] == [D, D']
        @test [h[1] for h in M.sideheaders] == [T, T'] && M.sideheaders[1][2] > M.sideheaders[2][2]   # top down
        @event A B
        @test_throws ArgumentError PPd.mosaic_layout([A, B]; facts=(P(A) => 0.3,))   # undetermined

        # headers: the 1% column's moved inside the figure and clear of its neighbour's;
        # the box is wide enough for the caption
        hx, sy, bbox = PPd._mosaicplaces(M, 0.13)
        hw = [PPd._textwidth(sprint(show, a), PPd.MOSAIC_FONT.header) for (a, _) in M.headers]
        @test hx[1] - hw[1] / 2 >= 0 && hx[1] + hw[1] / 2 < hx[2] - hw[2] / 2
        @test issorted(sy; rev=true) && bbox[2] > 1
        @test bbox[2] - bbox[1] >= PPd._textwidth(M.caption, PPd.MOSAIC_FONT.caption)
        @test PPd._tilefont("a long label for a thin tile", 0.05, 0.5, 0.04) == 0
        @test PPd._tilefont("Dᶜ ∩ Tᶜ", 0.9, 0.9, 0.04) == 0.04
    end

    @testset "sample events" begin
        rng = Xoshiro(3)
        draws = (σ=rand(rng, 1000) .* 0.5, μ=randn(rng, 1000))
        se = PPd.@sampleevents draws begin
            A = σ > 0.2
            B = μ > 0
            Cm = 0 < μ < 1
            Dm = (σ < 0.1) || (μ < -1)
        end
        @test A isa RV && PPd.isevent(A) && PPd.ndraws(se) == 1000
        @test se.masks[1] == (draws.σ .> 0.2)
        @test se.masks[3] == ((draws.μ .> 0) .& (draws.μ .< 1))
        @test se.masks[4] == ((draws.σ .< 0.1) .| (draws.μ .< -1))
        @test se.conditions[1] == :(σ > 0.2)
        a, b = draws.σ .> 0.2, draws.μ .> 0
        @test compute(P(A | B), se) ≈ count(a .& b) / count(b)
        @test compute(P(A ∪ B'), se) ≈ count(a .| .!b) / 1000
        @test compute(P(A) + P(B) - P(A ∩ B), se) ≈ compute(P(A ∪ B), se)
        evs, probs = cell_probabilities(se; events=[A, B])
        @test evs == [A, B] && sum(probs) ≈ 1 && probs[4] ≈ count(a .& b) / 1000

        # columns from a Dict; ordinary variables when no column has the name
        cut = 0.3
        se2 = PPd.@sampleevents Dict("σ" => draws.σ) begin
            E = σ > cut
        end
        @test se2.masks[1] == (draws.σ .> 0.3)
        se3 = PPd.sample_events(A => a, B => b)
        @test compute(P(A | B), se3) ≈ compute(P(A | B), se)
        @test_throws ArgumentError PPd.sample_events(A => a, B => b[1:10])

        # Venn, tree and mosaic numbers from the draws
        L = venn_layout(P(A | B); samples=se)
        @test PPd.venn_value(L, only(L.panels)) ≈ compute(P(A | B), se)
        @test L.conditions == Any[:(σ > 0.2), :(μ > 0)]
        @test PPd.tree_layout(P(A | B); samples=se).result ≈ compute(P(A | B), se)
        @test PPd.mosaic_layout(P(A | B); order=[B, A], samples=se).result ≈ compute(P(A | B), se)
        @test occursin("σ > 0.2", PPd.tree_layout([A, B]; samples=se).legend)
    end

    @testset "Venn numbers from facts" begin
        @event A B C
        facts = (P(A) => 0.3, P(B) => 0.5, P(A ∩ B) => 0.1)
        L = venn_layout(Equation(P(A ∪ B), inclusion_exclusion(P(A ∪ B))); facts)
        @test [PPd.venn_value(L, p) for p in L.panels] ≈ [0.7, 0.3, 0.5, 0.1]
        # the sets' order need not match the facts'
        L2 = venn_layout(P(B | A); sets=[B, A], facts)
        @test PPd.venn_value(L2, only(L2.panels)) ≈ 0.1 / 0.3
        @test L2.cellprobs[1+0b01] ≈ 0.4                 # B only (bit 0 = B)
        L3 = venn_layout(P(A ∩ C); facts)
        @test PPd.venn_value(L3, only(L3.panels)) === nothing   # C is not in the facts
        @test venn_layout(P(A)).cellprobs === nothing
        # a cell's label sits inside that cell
        for n in 1:3, c in 0:(1<<n)-1
            @test PPd._venn_cell(n, PPd._venn_anchor(n, c)...) == c
        end
    end

    if Base.find_package("CairoMakie") !== nothing
        @eval using CairoMakie: CairoMakie
        @testset "drawing" begin
            @rv A B C X Y Z
            @event D T E F
            conf = @dag begin
                Z → (X, Y)
                X → Y
            end
            dir = mktempdir()
            save(name, fig) = (f = joinpath(dir, name * ".png"); CairoMakie.save(f, fig); filesize(f))
            @test save("dag", PPd.dagplot(conf)) > 1000
            @test save("dsep", PPd.dagplot(conf, X ⟂ Y | Z)) > 1000
            @test save("do", PPd.dagplot(conf, P(Y | Do(X)))) > 1000
            test = BayesNet(D => 0.01, (T | D) => [0.95, 0.05])
            @test save("tree", PPd.probtree(P(D | T); bn=test)) > 1000
            @test save("mosaic", PPd.mosaic(P(D | T); bn=test)) > 1000
            f2 = compute(P(X | Z), rand_bayesnet(conf; rng=Xoshiro(1)))
            @test save("factor2", PPd.factorplot(f2)) > 1000
            @test save("factor2h", PPd.factorplot(f2; kind=:heatmap)) > 1000
            @test save("factor1", PPd.factorplot(compute(P(X), rand_bayesnet(conf; rng=Xoshiro(1))))) > 1000
            @test save("factor3", PPd.factorplot(compute(P(X, Y | Z), rand_bayesnet(conf; rng=Xoshiro(1))))) > 1000
            fig = CairoMakie.Figure()
            @test CairoMakie.barplot(fig[1, 1], compute(P(X), rand_bayesnet(conf; rng=Xoshiro(1)))) isa CairoMakie.Makie.AxisPlot
            @test CairoMakie.heatmap(fig[1, 2], f2) isa CairoMakie.Makie.AxisPlot
            facts = (P(E) => 0.3, P(F) => 0.5, P(E ∩ F) => 0.1)
            @test save("vennfacts", PPd.venn(Equation(P(E ∪ F), inclusion_exclusion(P(E ∪ F))); facts)) > 1000
            se = PPd.sample_events(E => rand(Xoshiro(2), 200) .> 0.5, F => rand(Xoshiro(3), 200) .> 0.3)
            @test save("vennsamples", PPd.venn(P(E | F); samples=se, cellvalues=:all)) > 1000
            ax = CairoMakie.Axis(fig[2, 1])
            @test PPd.dagplot!(ax, conf, X ⟂ Y | Z) isa PPd.DAGView
            @test PPd.probtree!(CairoMakie.Axis(fig[2, 2]), [D, T]; bn=test) isa PPd.TreeLayout
            # Select explicitly: earlier PBP tests deliberately pin their backend.
            oldbackend = PPd.plotbackend()
            PPd.plotbackend!(:makie)
            try
                chain = PPd.MarkovChain([3//4 1//4; 1//3 2//3]; states=[:sun, :rain])
                @test save("markov", PPd.markovplot(chain)) > 1000
                @test save("hmm", PPd.hmmplot(X, Y, 4)) > 1000
                @test PPd.markovplot!(CairoMakie.Axis(fig[3, 1]), chain) isa PPd.MarkovView
                @test PPd.hmmplot!(CairoMakie.Axis(fig[3, 2]), X, Y, 3) isa PPd.DAGView
            finally
                PPd.plotbackend!(oldbackend)
            end
        end
    else
        @info "CairoMakie not in the test environment; skipping diagram drawing tests"
    end
end

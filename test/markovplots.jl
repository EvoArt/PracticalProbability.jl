@testset "Markov and HMM plots" begin
    PP = PracticalProbability
    backend = plotbackend()
    plotbackend!(:pbp)
    mc = MarkovChain([1//2 1//2 0; 1//4 1//2 1//4; 0 0 1]; states=[:sun, :rain, :end])
    v = markov_view(mc)
    @test length(v.transitions) == 6
    @test count(e -> e.from == e.to, v.transitions) == 3
    @test v.absorbing == [false, false, true]
    lowerloop = only(filter(e -> e.from == 3 && e.to == 3, v.transitions))
    @test lowerloop.labelpos[2] < v.positions[3][2]  # outward, clear of the incoming edge
    @test only(filter(e -> e.from==2 && e.to==1, v.transitions)).probability == 1//4
    @test length(markov_view(mc; selfloops=false).transitions) == 3
    @test length(markov_view(mc; minprob=1//4).transitions) == 4
    @test length(markov_view(MarkovChain(reshape([1], 1, 1))).transitions) == 1
    @test_throws ArgumentError markov_view(mc; minprob=-1)
    @test_throws ArgumentError markov_view(mc; positions=[(0,0)])
    @test_throws ArgumentError markov_view(mc; positions=fill((0,0), 3))

    @rv X Y
    hv = hmm_view(X, Y, 4; observed=[true, false, true, true])
    panel = only(hv.panels)
    @test length(panel.nodes) == 8
    @test length(panel.edges) == 7
    @test count(n -> n.fill === :observed, panel.nodes) == 3
    @test all(e -> (e.from, e.to) in [(X[t], X[t+1]) for t in 1:3] ||
                  (e.from, e.to) in [(X[t], Y[t]) for t in 1:4], panel.edges)
    @test all(n.y == 0 for n in panel.nodes if n.v in X[1:4])
    @test_throws ArgumentError hmm_view(X, Y, 0)
    @test_throws ArgumentError hmm_view(X, X, 2)
    @test_throws ArgumentError hmm_view(X, Y, 2; observed=[true])
    bn = BayesNet(X[1]=>[0.5,0.5], (X[2]|X[1])=>[0.7 0.2; 0.3 0.8],
                  (Y[1]|X[1])=>[0.9 0.1; 0.1 0.9], (Y[2]|X[2])=>[0.9 0.1; 0.1 0.9])
    @test length(only(hmm_view(bn, X[1:2], Y[1:2]).panels).edges) == 3
    @test_throws ArgumentError hmm_view(bn, reverse(X[1:2]), Y[1:2])

    for fig in (markovplot(mc), markovplot(mc; labels=false), hmmplot(X,Y,4), hmmplot(bn,X[1:2],Y[1:2]))
        svg = PP.PBP.svg(fig)
        @test occursin("<svg", svg) && !occursin("NaN", svg)
        @test occursin("tikzpicture", PP.PBP.tikz(fig))
        @test occursin("cetz", PP.PBP.cetz(fig))
    end
    fig = PP.PBP.Figure()
    ax = PP.PBP.Axis(fig[1,1]; aspect=PP.PBP.DataAspect())
    @test markovplot!(ax, mc) isa MarkovView
    @test hmmplot!(ax, X,Y,3) isa DAGView
    @test plotbackend!(backend) == backend
end

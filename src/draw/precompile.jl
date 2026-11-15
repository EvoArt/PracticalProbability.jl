# Draw each kind of diagram once while the package is precompiled, so the
# first diagram in a session is fast (as PracticalBayesPlot does for its
# plots).

function _draw_workload()
    @event A B C
    @rv X Y Z
    g = DAG(Z => X, Z => Y, X => Y)
    test = BayesNet(A => 0.01, (B | A) => [0.95, 0.05])
    bn = BayesNet(X => [0.3, 0.7], (Y | X) => [0.9 0.2; 0.1 0.8])
    figs = Any[
        _dagplot(Val(:pbp), g),
        _dagplot(Val(:pbp), g, P(Y | Do(X))::Prob),
        _dagplot(Val(:pbp), DAG(X => Y, Y => Z), X ⟂ Z | Y),
        _probtree(Val(:pbp), P(A | B); bn=test),
        _probtree(Val(:pbp), [A, B]),
        _mosaic(Val(:pbp), P(A | B); bn=test),
        _venn(Val(:pbp), P(A | B)),
        _venn(Val(:pbp), P(A ∩ B | C); labels=:text),
        _venn(Val(:pbp), Equation(P(A | B), bayes(P(A | B)::Prob))),
        _factorplot(Val(:pbp), compute(P(Y | X), bn)),
        _factorplot(Val(:pbp), compute(P(Y | X), bn); kind=:heatmap),
        _factorplot(Val(:pbp), compute(P(X), bn)),
    ]
    io = IOBuffer()
    for f in figs
        show(io, MIME"image/svg+xml"(), f)
        PBP.tikz(io, f)
    end
    PBP.CURRENT[] = nothing
    return nothing
end

if ccall(:jl_generating_output, Cint, ()) == 1
    @static if VERSION >= v"1.12.0-"
        ccall(:jl_tag_newly_inferred_enable, Cvoid, ())
        try
            _draw_workload()
        finally
            ccall(:jl_tag_newly_inferred_disable, Cvoid, ())
        end
    else
        _draw_workload()
    end
end

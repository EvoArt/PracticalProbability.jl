@testset "terms" begin
    @rv A B C θ::continuous
    @test A isa RV && isdiscrete(A) && iscontinuous(θ)

    p = P(A, B | C)
    @test p.targets == [A, B] && p.given == [C]
    @test P(B, A | C) == p                       # arguments are sets
    @test hash(P(B, A | C)) == hash(p)
    @test P(A | B, C) == P(A | C, B)
    @test P(A, B | C) != P(A | B, C)
    @test_throws ArgumentError P(Do(A))          # do(⋅) only after the bar
    @test_throws ArgumentError P(A | B, C | B)   # one bar

    # events and interventions
    e = P(A(1) | Do(B), C(:x))
    @test e.targets[1] isa Event && e.given[1] isa Intervention
    @test sprint(show, e) == "P(A=1 | do(B), C=x)"
    @test sprint(show, P(θ | A)) == "p(θ | A)"   # densities for continuous targets

    # the macro declares as it goes, and reuses bound RVs
    q = @P(X, Y | Z)
    @test sprint(show, q) == "P(X, Y | Z)"
    @test @P(θ | A) == P(θ | A)
    @test @P(A(2) | Do(B)) == P(A(2) | Do(B))

    # products are multisets, constants fold, sums merge
    @test P(A) * P(B) == P(B) * P(A)
    @test 1 * P(A) == P(A)
    @test (P(A) * P(B)) * P(C) isa Product && length(((P(A) * P(B)) * P(C)).factors) == 3
    @test Σ(A, Σ(B, P(A, B))) == Σ([A, B], P(A, B))
    @test P(A) / Literal(1) == P(A)

    @test freevars(Σ(B, P(A, B | C))) == [A, C]
    @test freevars(P(A(1) | B)) == [B]
    @test substitute(P(A | B), B => B(2)) == P(A | B(2))
    @test substitute(Σ(B, P(A, B)), B => B(1)) == Σ(B, P(A, B))   # bound variable untouched
end

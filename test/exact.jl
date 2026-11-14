@testset "exact arithmetic" begin
    @rv X Y
    @event A B
    C_ = RV(:C)

    @testset "networks with rational tables" begin
        bn = BayesNet(X => [1//3, 2//3], (Y | X) => [1//2 1//4; 1//2 3//4])
        v = compute(P(X(1) | Y(1)), bn)
        @test v isa Rational && v == 1//2           # (1/6) / (1/6 + 1/6)
        @test compute(P(Y(2)), bn) === 2//3
        f = compute(P(X | Y), bn)
        @test eltype(f.values) <: Rational
        # integer tables are exact too; floats stay floats
        bn2 = BayesNet(X => [1, 0], (Y | X) => [0 1; 1 0])
        @test compute(P(Y(2)), bn2) === 1//1
        @test compute(P(X(1)), BayesNet(X => [0.3, 0.7])) isa Float64
        # events from a probability, exactly
        bn3 = BayesNet(A => 1//100, (B | A) => [9//10, 1//20])
        @test compute(P(A | B), bn3) == 2//13
        # NaN for conditioning on an impossible event, as with floats
        @test isnan(compute(P(Y(1) | X(2)), bn2))

    end

    @testset "random exact networks: identities hold with ==" begin
        g = @dag X → C_ → Y
        nb = rand_bayesnet(g; rng=RNG, exact=true)
        @test eltype(cpt(nb, Y).values) <: Rational
        @test verify(P(Y | X, C_), P(Y | C_), nb; atol=0)
        @test verify(Σ(C_, P(Y | C_) * P(C_ | X)), P(Y | X), nb; atol=0)
        lhs, rhs = compute(Equation(P(X | Y), bayes(P(X | Y))), nb)
        @test lhs.values == PracticalProbability._align(rhs, lhs.vars)
    end

    @testset "literals and printing" begin
        @test (Literal(1) / Literal(3)).value === 1//3
        @test sprint(show, Literal(1//3)) == "1/3"
        @test sprint(show, Literal(4//2)) == "2"
        @test sprint(show, (1//3) * P(A)) == "1/3 P(A)"
    end

    @testset "deduce in rational arithmetic" begin
        s = deduce(P(A | B), P(B | A) => 9//10, P(A) => 1//100, P(B | A') => 1//20)
        @test s.determined && s.value === 2//13
        @test sprint(show, s) == "P(A | B) = 2/13"
        @test deduce(P(A ∩ B), P(A) => 3//10, P(B) => 1//2, P(A ∪ B) => 3//5).value === 1//5
        @test deduce(P(A ∪ B), P(A) => 3//10, P(B) => 2//5, A ⟂ B).value === 29//50
        # floats as before
        @test deduce(P(A ∪ B), P(A) => 0.3, P(B) => 0.4, A ⟂ B).value isa Float64
        @test !deduce(P(A ∩ B), P(A) => 1//3, P(B) => 1//2).determined
        @test_throws ArgumentError deduce(P(A), P(A) => 1//3, P(A') => 1//3)
        evs, probs = cell_probabilities(P(A) => 1//2, P(B) => 1//2, A ⟂ B)
        @test all(==(1//4), probs)
    end
end

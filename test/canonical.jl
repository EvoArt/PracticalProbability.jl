@testset "canonical forms" begin
    @rv X Y Z a b c
    @rv A B C D
    RConst = PracticalProbability.RConst

    @testset "expand" begin
        @test expand((X + Y)^2) == X^2 + 2X * Y + Y^2
        @test sprint(show, expand((X + Y)^2)) == "X² + 2X Y + Y²"
        @test expand((X + 1) * (X - 1)) == X^2 - 1
        @test expand((X + Y) * (X - Y)) == X^2 - Y^2
        @test expand(exp(2 * (X + 1))) == exp(2X + 2)
        @test expand(X - X) == RConst(0)
        # equal polynomials print the same whatever order they were built in
        @test sprint(show, expand(Y * X + X^2)) == sprint(show, expand(X^2 + X * Y))
    end

    @testset "lowest terms" begin
        @test simplify((a * b + a * c) / a) == expand(b + c)
        @test simplify((X^2 - Y^2) / (X - Y)) == expand(X + Y)
        @test simplify(X / X) == RConst(1)
        @test simplify((X^2 + 2X * Y + Y^2) / (X + Y)) == expand(X + Y)
        r = simplify((X + 1) / (X + 2))            # nothing cancels
        @test evaluate(r, X => 3) ≈ 4 / 5
        @test simplify(2X / 4) == expand(X / 2)
    end

    @testset "derivatives and substitution" begin
        @test derivative(X^3, X) == 3 * X^2
        @test derivative(exp(2X), X) == 2 * exp(2X)
        @test evaluate(derivative(log(X^2 + 1), X), X => 2.0) ≈ 4 / 5
        @test derivative(X^2 * Y, Y) == X^2
        @test derivative(X^3, X, 2) == 6X
        @test derivative(Y, X) == RConst(0)
        @test evaluate(derivative(sqrt(X), X), X => 4.0) ≈ 1 / 4
        @test evaluate(derivative(atan(X), X), X => 1.0) ≈ 1 / 2
        @test substitute(exp(X * Y), Y => 0) == RConst(1)
        @test substitute(X^2 + Y, X => Y) == Y^2 + Y
        @test evaluate(min(X, Y) + max(X, 2), X => 1, Y => 3) == 3
    end

    @testset "probability expressions" begin
        @test sprint(show, canonical(P(B, A | D, C))) == "P(A, B | C, D)"
        @test sprint(show, canonical(P(B | C) * P(A))) == "P(A) P(B | C)"
        e1 = P(C) + P(A | B) * P(B)
        e2 = P(B) * P(A | B) + P(C)
        @test e1 == e2
        @test sprint(show, canonical(e1)) == sprint(show, canonical(e2))
        @test canonical(E(Y * X + X | Z)) == canonical(E(X + X * Y | Z))
        @test sprint(show, canonical(Cov(Y, X))) == "Cov(X, Y)"
    end

    @testset "simplify by search" begin
        g = @dag A → B → C
        bn = rand_bayesnet(g; rng=RNG)
        for e in (bayes(P(A | B)), Σ(B, P(C | B) * P(B | A)), P(A | B) * P(B) / P(A, B) + P(C | A, B))
            greedy = simplify(e, g)
            found = simplify(e, g; search=true)
            @test PracticalProbability._esize(found) <= PracticalProbability._esize(greedy)
            @test verify(e, found, bn)
        end
        d = simplify(bayes(P(A | B)); search=true, trace=true)
        @test d isa Derivation && last(d) == P(A | B)
    end
end

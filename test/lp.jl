@testset "bounds by linear programming" begin
    @event A B
    @rv X Y Z

    @testset "deduce gives bounds" begin
        s = deduce(P(A ∩ B), P(A) => 3//10, P(B) => 1//2)
        @test !s.determined && extrema(s) == (0, 3//10)           # Fréchet
        @test occursin("between 0 and 3/10", sprint(show, s))
        @test extrema(deduce(P(A ∪ B), P(A) => 3//10, P(B) => 1//2)) == (1//2, 4//5)
        @test extrema(deduce(P(A | B), P(A) => 3//10, P(B) => 1//2)) == (0, 3//5)
        lo, hi = extrema(deduce(P(A ∩ B), P(A) => 0.7, P(B) => 0.6))
        @test lo ≈ 0.3 && hi ≈ 0.6
        @test extrema(deduce(P(A), P(A) => 1//4)) == (1//4, 1//4)
    end

    @testset "the simplex method" begin
        # min -x₁ - x₂, x₁ + 2x₂ + s₁ = 4, 3x₁ + x₂ + s₂ = 6
        st, v, x = PracticalProbability._simplex([-1, -1, 0, 0], [1 2 1 0; 3 1 0 1], [4, 6])
        @test st === :optimal && v == -14//5 && x[1:2] == [8//5, 6//5]
        @test PracticalProbability._simplex([1, 1], [1 1], [-1])[1] === :infeasible
        @test PracticalProbability._simplex([-1, 0], [1 -1], [0])[1] === :unbounded
        # redundant rows are fine
        @test PracticalProbability._simplex([1, 2], [1 1; 2 2], [1, 2])[2] == 1
    end

    nat = @admg begin
        X → Y
        X ↔ Y
    end
    @testset "natural bounds" begin
        bn = BayesNet(X => [1//3, 2//3], (Y | X) => [1//4 1//2; 3//4 1//2])
        b = lp_bounds(P(Y(1) | Do(X(1))), nat, bn)
        @test extrema(b) == (1//12, 3//4)                         # [P(x, y), P(x, y) + 1 - P(x)]
        ace = lp_bounds(P(Y(1) | Do(X(1))) - P(Y(1) | Do(X(2))), nat, bn)
        @test ace.upper - ace.lower == 1
        @test occursin("≤", sprint(show, b))
        # a randomised treatment: identified, the bounds meet
        rct = @dag X → Y
        @test extrema(lp_bounds(P(Y(1) | Do(X(1))), rct, bn)) == (1//4, 1//4)
    end

    @testset "instrumental-variable bounds" begin
        iv = @admg begin
            Z → X → Y
            X ↔ Y
        end
        for _ in 1:4
            bn = rand_bayesnet(canonical_dag(iv); rng=RNG, exact=true)
            for x in 1:2
                q = P(Y(1) | Do(X(x)))
                truth = compute(q, bn)
                b = lp_bounds(q, iv, bn)
                bnat = lp_bounds(q, nat, bn)
                @test b.lower <= truth <= b.upper
                @test bnat.lower <= b.lower && b.upper <= bnat.upper
                @test b.lower isa Rational
            end
        end
        @test_throws ArgumentError lp_bounds(P(Y(1) | X(1)), iv, rand_bayesnet(canonical_dag(iv); rng=RNG))
    end
end

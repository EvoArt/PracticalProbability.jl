@testset "assumptions and deduce" begin
    @event A B C
    @rv X

    @testset "simplify with facts" begin
        f = assume(A ⟂ B)
        @test f isa Facts && !isempty(f)
        @test simplify(P(A ∩ B), f) == P(A) * P(B)
        @test simplify(P(A | B), f) == P(A)
        @test simplify(P(A' | B), f) == P(A')
        @test simplify(P(A ∩ B'), f) == P(A) * P(B')
        @test simplify(P(A | B), assume(A ⟂ B | C)) == P(A | B)       # needs C given
        @test simplify(P(A | B, C), assume(A ⟂ B | C)) == P(A | C)
        @test simplify(P(A | B, C), assume(A ⟂ (B, C))) == P(A)        # weak union
        @test simplify(inclusion_exclusion(P(A ∪ B)), f) == P(A) + P(B) - P(A) * P(B)
        @test simplify(P(A ∩ B), assume(A ⟂ C)) == P(A ∩ B)             # nothing assumed about B

        d = assume(disjoint(A, B))
        @test simplify(P(A ∩ B), d) == Literal(0)
        @test simplify(P(A | B), d) == Literal(0)
        @test simplify(P(A ∪ B | C), d) == P(A | C) + P(B | C)
        @test simplify(inclusion_exclusion(P(A ∪ B)), d) == P(A) + P(B)

        s = assume(implies(A, B))
        @test simplify(P(A ∩ B), s) == P(A)
        @test simplify(P(B | A), s) == Literal(1)
        @test simplify(P(A ∪ B), s) == P(B)
        @test simplify(P(A' | B'), s) == Literal(1)                    # Bᶜ ⊆ Aᶜ
        d2 = simplify(P(A ∩ B), s; trace=true)
        @test occursin("(assumed)", d2.rules[end])

        @test_throws ArgumentError assume(42)
        @test occursin("A ⟂ B", sprint(show, MIME"text/plain"(), f))
    end

    @testset "facts hold numerically where they are true" begin
        g = @dag C → (A, B)                  # A ⟂ B | C
        bn = rand_bayesnet(g; rng=RNG)
        fc = assume(A ⟂ B | C)
        for e in (P(A | B, C), P(A ∩ B | C), P(A' ∩ B | C), P(A ∪ B | C))
            @test verify(simplify(e, fc; trace=true), bn)
        end
        nb = BayesNet(A => 0.3, (B | A) => [0.0, 0.4])   # B never happens with A
        @test verify(P(A ∪ B), simplify(P(A ∪ B), assume(disjoint(A, B))), nb)
        ib = BayesNet(A => 0.3, (B | A) => [1.0, 0.4])   # A implies B
        for e in (P(A ∩ B), P(A ∪ B), P(A' | B'))
            @test verify(e, simplify(e, assume(implies(A, B))), ib)
        end
    end

    @testset "deduce" begin
        s1 = deduce(P(A ∩ B), P(A) => 0.3, P(B) => 0.5, P(A ∪ B) => 0.6)
        @test s1 isa Solution && s1.determined && Float64(s1) ≈ 0.2
        @test sprint(show, s1) == "P(A ∩ B) = 0.2"
        @test Float64(deduce(P(A | B), P(A) => 0.3, P(B) => 0.5, P(A ∪ B) => 0.6)) ≈ 0.4
        @test Float64(deduce(P(A ∪ B), P(A) => 0.3, P(B) => 0.4, A ⟂ B)) ≈ 0.58
        @test Float64(deduce(P(A' ∩ B'), P(A) => 0.3, P(B) => 0.4, A ⟂ B)) ≈ 0.42
        @test Float64(deduce(P(A | B), P(B | A) => 0.9, P(A) => 0.01, P(B | A') => 0.05)) ≈
              0.009 / (0.009 + 0.99 * 0.05)
        @test Float64(deduce(P(A ∪ B), P(A) => 0.2, P(B) => 0.3, disjoint(A, B))) ≈ 0.5
        @test Float64(deduce(P(B), P(A) => 0.2, implies(A, B), P(B ∩ A') => 0.1)) ≈ 0.3
        @test Float64(deduce(1 - P(A) + P(A ∩ B), P(A) => 0.4, P(A ∩ B) => 0.1)) ≈ 0.7
        # conditional independence: products follow once the parts are known
        v = deduce(P(A ∩ B), A ⟂ B | C, P(C) => 0.5, P(A | C) => 0.2, P(A | C') => 0.6,
                   P(B | C) => 0.3, P(B | C') => 0.7)
        @test Float64(v) ≈ 0.5 * 0.2 * 0.3 + 0.5 * 0.6 * 0.7

        u = deduce(P(A ∩ B), P(A) => 0.3, P(B) => 0.5)
        @test !u.determined && isnan(Float64(u)) && occursin("not determined", sprint(show, u))
        @test_throws ArgumentError deduce(P(A), P(A) => 0.3, P(A) => 0.4)    # inconsistent
        @test_throws ArgumentError deduce(P(X(1)), P(X(1)) => 0.3)             # not an event

        ev, cells = cell_probabilities(P(A) => 0.3, P(B) => 0.4, A ⟂ B; events=[A, B])
        @test ev == [A, B] && length(cells) == 4
        @test cells[1] ≈ 0.42 && cells[2] ≈ 0.18 && cells[3] ≈ 0.28 && cells[4] ≈ 0.12
        _, open = cell_probabilities(P(A) => 0.3, P(B) => 0.5)
        @test all(isnothing, open)
    end
end

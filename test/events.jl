@testset "events" begin
    @event A B C
    @rv X Y

    @testset "building events" begin
        @test isevent(A) && iseventlike(A) && iseventlike(X(1)) && !iseventlike(X)
        @test A ∪ B == B ∪ A && A ∩ B == B ∩ A
        @test (A ∪ B) ∪ C == A ∪ (B ∪ C)
        @test (A')' == A && !A == A' && complement(A) == A'
        @test A ∪ A == A && A ∩ A == A
        @test A ∪ A' == Ω && A ∩ A' == ∅
        @test A ∩ Ω == A && A ∪ ∅ == A && A ∪ Ω == Ω && A ∩ ∅ == ∅
        @test Ω' == ∅ && ∅' == Ω
        @test setdiff(A, B) == A ∩ B'
        @test rvs(A ∪ X(1)) == [A, X]
        @test_throws ArgumentError X ∪ Y          # plain variables are not events
        @test_throws ArgumentError X'

        @test P(A ∩ B | C) == P(A, B | C)         # intersection = listing the targets
        @test P(A | B ∩ C) == P(A | B, C)
        @test P(Ω) == Literal(1) && P(∅ | A) == Literal(0) && P(A ∩ A') == Literal(0)
        @test P(A | Ω) == P(A)
        @test_throws ArgumentError P(A | ∅)
        @test P(A ∪ B | C).targets == [A ∪ B]
        @test P(A | B ∪ C) == P(A | (B ∪ C))           # ∪ after the bar means the condition
        @test P(A | C, B ∪ C) == P(A | C, (B ∪ C))
    end

    @testset "display" begin
        @test sprint(show, P(A ∪ B | C)) == "P(A ∪ B | C)"
        @test sprint(show, P(A ∩ B)) == "P(A ∩ B)"
        @test sprint(show, P(A')) == "P(Aᶜ)"
        @test sprint(show, P((A ∪ B)')) == "P((A ∪ B)ᶜ)"
        @test sprint(show, P(A ∩ (B ∪ C))) == "P(A ∩ (B ∪ C))"
        @test sprint(show, P(X(1) ∪ Y(2))) == "P({X=1} ∪ {Y=2})"
        @test sprint(show, P(X(1), Y(2))) == "P(X=1, Y=2)"
        @test sprint(show, 1 - P(A)) == "1 - P(A)"
        @test sprint(show, P(A) + P(B) - P(A ∩ B)) == "P(A) + P(B) - P(A ∩ B)"
        @test latex(P(A ∪ B'); env=:raw) == raw"P(A \cup B^{\mathsf{c}})"
        @test latex(P(A ∩ B | C); env=:raw) == raw"P(A \cap B \mid C)"
        @test latex(1 - P(A); env=:raw) == "1 - P(A)"
        @test latex(P(Ω); env=:raw) == "1"
    end

    @testset "@hold" begin
        held = @hold (P(A ∩ A'), P(A ∪ A'), P(A ∩ Ω), P(A ∪ ∅), P((A')'), P(A ∩ A), P(Ω), P(A, A'))
        @test map(e -> sprint(show, e), held) ==
              ("P(A ∩ Aᶜ)", "P(A ∪ Aᶜ)", "P(A ∩ Ω)", "P(A ∪ ∅)", "P((Aᶜ)ᶜ)", "P(A ∩ A)", "P(Ω)", "P(A ∩ Aᶜ)")
        @test map(simplify, held) == (Literal(0), Literal(1), P(A), P(A), P(A), P(A), Literal(1), Literal(0))
        @test sprint(show, Equation(held[1], simplify(held[1]))) == "P(A ∩ Aᶜ) = 0"
        @test simplify(held[1]; trace=true).rules[2:end] == ["A ∩ Aᶜ = ∅", "P(∅) = 0"]
        @test simplify(held[2]; trace=true).rules[2:end] == ["A ∪ Aᶜ = Ω", "P(Ω) = 1"]
        @test simplify(held[5]; trace=true).rules[2:end] == ["(Aᶜ)ᶜ = A"]
        # nested: one identity per step
        e = @hold P(B | (A ∪ ∅) ∩ (C')')
        @test length(simplify(e; trace=true)) == 3 && simplify(e) == P(B | A, C)
        # a canonical intersection is just the joint targets; nothing to do
        @test @hold(P(A ∩ B | C)) == P(A, B | C)
        @test simplify(@hold(P(A ∩ B ∩ Ω)); trace=true).rules[2:end] == ["A ∩ B ∩ Ω = A ∩ B"]
        @test simplify(@hold(P(A | B ∪ Ω))) == P(A)
        @test_throws ArgumentError simplify(@hold P(A | B ∩ B'))
        @test_throws ArgumentError @hold P(A | ∅)
        # only construction is held, and the flag does not leak out of the block
        @test @hold(simplify(P(A ∩ A'))) == Literal(0)
        @test A ∩ A' == ∅ && P(A ∩ A') == Literal(0)
        @test latex(@hold(P((A')')); env=:raw) == raw"P((A^{\mathsf{c}})^{\mathsf{c}})"
        @test simplify(@hold(P(A ∩ B ∩ Ω)), assume(disjoint(A, B))) == Literal(0)
    end

    g = @dag begin
        A → (B, C)
        B → C
    end
    bn = rand_bayesnet(g; rng=RNG)

    @testset "event networks" begin
        @test domain(bn, A) == [true, false]
        small = BayesNet(A => 0.3, (B | A) => [0.9, 0.2])
        @test compute(P(A), small) ≈ 0.3 && compute(P(A'), small) ≈ 0.7
        @test compute(P(B | A'), small) ≈ 0.2
        @test compute(P(B), small) ≈ 0.3 * 0.9 + 0.7 * 0.2
        @test compute(P(A ∪ B), small) ≈ 1 - 0.7 * 0.8
        @test compute(P(A | B'), small) ≈ 0.3 * 0.1 / (0.3 * 0.1 + 0.7 * 0.8)
        @test compute(P(Ω), small) == 1
    end

    @testset "complements" begin
        e = complement_rule(P(A | C))
        @test e == 1 - P(A' | C)
        @test verify(P(A | C), e, bn)
        @test simplify(e) == P(A | C)
        @test verify(P(A ∩ B), complement_rule(P(A ∩ B)), bn)
        @test simplify(1 - P(B | A)) == P(B' | A)
        @test simplify(P(A' | A)) == Literal(0)
        @test simplify(P(A ∪ B | A)) == Literal(1)
        @test simplify(P(A | C) + P(A' | C)) == Literal(1)
        @test simplify(P(A ∩ B) + P(A ∩ B')) == P(A)
        @test simplify(P(A | C) * P(B) + P(A' | C) * P(B)) == P(B)   # common factor
        # P(C | A) + P(C | Aᶜ) is not 1: nothing to do
        @test simplify(P(C | A) * P(B) + P(C | A') * P(B)) == P(C | A) * P(B) + P(C | A') * P(B)
    end

    @testset "De Morgan" begin
        @test demorgan((A ∪ B)') == A' ∩ B'
        @test demorgan((A ∩ B)') == A' ∪ B'
        @test demorgan((A ∪ (B ∩ C))') == A' ∩ (B' ∪ C')
        @test demorgan(P((A ∪ B)' | C)) == P(A', B' | C)
        @test verify(P((A ∪ B)'), demorgan(P((A ∪ B)')), bn)
        @test verify(P((A ∩ B)' | C), demorgan(P((A ∩ B)' | C)), bn)
    end

    @testset "inclusion–exclusion" begin
        @test inclusion_exclusion(P(A ∪ B)) == P(A) + P(B) - P(A ∩ B)
        for q in (P(A ∪ B), P(A ∪ B | C), P(A ∪ B ∪ C), P((A ∪ B)'), P(C ∩ (A ∪ B)),
                  P(A ∪ B' | C), P(X(1) ∪ Y(2)))
            nets = q == P(X(1) ∪ Y(2)) ? rand_bayesnet(@dag(X → Y); card=3, rng=RNG) : bn
            @test verify(q, inclusion_exclusion(q), nets)
        end
        @test length(inclusion_exclusion(P(A ∪ B ∪ C)).terms) == 7
        @test_throws ArgumentError inclusion_exclusion(P(A))
        @test compute(union_bound(P(A ∪ B)), bn) >= compute(P(A ∪ B), bn)
    end

    @testset "total probability and Bayes" begin
        e = total_probability(P(C), A)
        @test e == P(C | A) * P(A) + P(C | A') * P(A')
        @test verify(P(C), e, bn)
        @test simplify(e) == P(C)
        @test verify(P(C | B), total_probability(P(C | B), A), bn)
        @test verify(P(C), marginalize(P(C), A), bn)

        b = bayes(P(A | C); expand=true)
        @test b == P(C | A) * P(A) / (P(C | A) * P(A) + P(C | A') * P(A'))
        @test verify(P(A | C), b, bn)
        @test simplify(b) == P(A | C)
        # events of ordinary variables: X = 1 against X ≠ 1
        nb = rand_bayesnet(@dag(X → Y); card=3, rng=RNG)
        @test verify(P(X(2) | Y(1)), bayes(P(X(2) | Y(1)); expand=true), nb)
        @test compute(P(X(1)'), nb) ≈ 1 - compute(P(X(1)), nb)
    end

    @testset "expand on event graphs" begin
        for q in (P(C), P(A | C), P(C | A'), P(B' | C), P(A, C))
            d = expand(q, g; trace=true)
            @test verify(d, bn)
        end
        @test expand(P(A), g) == P(A)
        @test_throws ArgumentError expand(P(A ∪ B), g)
    end
end

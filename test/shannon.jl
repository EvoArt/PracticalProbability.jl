@testset "information inequalities" begin
    @rv X Y Z A B C D

    @testset "Shannon-type" begin
        p = prove(H(X, Y) <= H(X) + H(Y))
        @test p.holds && Bool(p)
        @test p.terms == [1 => MI(X, Y)]
        @test isshannon(MI(X, Y | Z) >= 0)
        @test isshannon(H(X | Y, Z) <= H(X | Y))                  # conditioning reduces entropy
        @test isshannon(H(X, Y, Z) <= H(X, Y) + H(Y, Z) - H(Y) + H(X | Y))
        @test isshannon(MI(X, (Y, Z)) >= MI(X, Y))
        @test isshannon(2 * H(X, Y, Z) <= H(X, Y) + H(Y, Z) + H(X, Z))   # Han's inequality, n = 3
        # the proof is an identity: rhs - lhs = ∑ λ e
        q = H(X, Y) + H(Y, Z) >= H(X, Y, Z) + H(Y)
        pr = prove(q)
        @test pr.holds
        bn = rand_bayesnet(DAG(X => Y, Y => Z, X => Z); rng=RNG)
        @test compute(q.rhs - q.lhs, bn) ≈ compute(proof_expr(pr), bn)
        @test occursin("proved", sprint(show, MIME"text/plain"(), pr))
    end

    @testset "with assumptions" begin
        dpi = MI(X, Z) <= MI(X, Y)
        @test !isshannon(dpi)
        p = prove(dpi; assume=[X ⟂ Z | Y])
        @test p.holds && !isempty(p.assumed)
        g = @dag X → Y → Z
        @test isshannon(dpi; dag=g)
        @test isshannon(data_processing(g, X, Y, Z); dag=g)
        @test isshannon(dpi; assume=assume(X ⟂ Z | Y))
        # a function of X has no more entropy than X
        @test isshannon(H(Y) <= H(X); assume=[H(Y | X)])
    end

    @testset "not Shannon-type" begin
        # false: X, Y independent fair bits and Z = X ⊻ Y
        p = prove(MI(X, Y | Z) <= MI(X, Y))
        @test !p.holds
        @test p.counterexample isa Factor && p.violation > 0
        joint = p.counterexample
        @test sum(joint.values) ≈ 1
        @test !prove(H(X) <= H(X | Y)).holds
        @test occursin("False", sprint(show, MIME"text/plain"(), p))
        # Zhang–Yeung: true, but not a consequence of Shannon's inequalities
        zy = 2 * MI(C, D) <= MI(A, B) + MI(A, (C, D)) + 3 * MI(C, D | A) + MI(C, D | B)
        pz = prove(zy)
        @test !pz.holds && pz.counterexample === nothing
        @test occursin("Not proved", sprint(show, MIME"text/plain"(), pz))
        # …but it follows from the copy lemma, and trivially from itself
        pc = prove(zy; copies=:auto)
        @test pc.holds && length(pc.copies) == 1
        @test occursin("copy lemma", sprint(show, MIME"text/plain"(), pc))
        @test prove(zy; nonshannon=true).holds
        # a consequence of Zhang–Yeung that is not Shannon-type: ZY with A, B swapped and doubled
        zy2 = 4 * MI(C, D) <= 2 * MI(B, A) + 2 * MI(B, (C, D)) + 6 * MI(C, D | B) + 2 * MI(C, D | A)
        pn = prove(zy2; nonshannon=true)
        @test pn.holds && !isempty(pn.nonshannon)
        @test !isshannon(zy2)
    end

    @testset "errors" begin
        @test_throws ArgumentError prove(P(X) <= H(X))
        @test_throws ArgumentError prove(H(X | Y(1)) >= 0)
    end
end

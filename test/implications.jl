@testset "testable implications" begin
    @rv A B C D V1 V2 V3 V4 X Y Z

    @testset "a basis of independences" begin
        chain = @dag A → B → C
        b = markov_basis(chain)
        @test length(b) == 1 && b[1].X == [C] && b[1].Y == [A] && b[1].Z == [B]
        bn = rand_bayesnet(chain; rng=RNG)
        @test all(s -> verify(P(s.X | (s.Z..., s.Y...)), P(s.X | (s.Z...,)), bn), b)
        iv = @admg begin
            Z → X → Y
            X ↔ Y
        end
        @test isempty(markov_basis(iv))           # the IV model implies no independence
        fd = @admg begin
            X → Z → Y
            X ↔ Y
            A → X
        end
        @test any(s -> s.X == [Z] && s.Y == [A], markov_basis(fd))     # Z ⟂ A | X
    end

    @testset "Verma constraints" begin
        verma = @admg begin
            V1 → V2 → V3 → V4
            V2 ↔ V4
        end
        cs = verma_constraints(verma)
        @test length(cs) == 1 && cs[1].noton == [V1] && issetequal(cs[1].on, [V3, V4])
        @test occursin("does not depend on V1", sprint(show, cs[1]))
        for _ in 1:3
            @test holds(cs[1], rand_bayesnet(canonical_dag(verma); rng=RNG))
        end
        # a direct V1 → V4 edge breaks it
        broken = ADMG(DAG(V1 => V2, V2 => V3, V3 => V4, V1 => V4), [(V2, V4)])
        @test !holds(cs[1], rand_bayesnet(canonical_dag(broken); rng=RNG))
        @test isempty(verma_constraints(@dag A → B → C))
        @test isempty(verma_constraints(@admg begin
            Z → X → Y
            X ↔ Y
        end))
    end

    @testset "Markov equivalence" begin
        chain = @dag A → B → C
        rev = @dag C → B → A
        fork = @dag B → (A, C)
        coll = DAG(A => B, C => B)
        @test markov_equivalent(chain, rev) && markov_equivalent(chain, fork)
        @test !markov_equivalent(chain, coll)
        @test vstructures(coll) == [(A, B, C)]
        @test cpdag(chain) == cpdag(fork)
        @test isempty(cpdag(chain).directed) && length(cpdag(chain).undirected) == 2
        @test cpdag(coll).directed == Set([A => B, C => B])
        @test length(equivalence_class(chain)) == 3
        @test length(equivalence_class(coll)) == 1
        # Meek rule 1 orients B → D below the collider
        g = DAG(A => B, C => B, B => D)
        @test (B => D) in cpdag(g).directed && length(equivalence_class(g)) == 1
        # equivalent DAGs give the same independences
        @test all(h -> issetequal(independencies(h), independencies(chain)), equivalence_class(chain))
        @test occursin("—", sprint(show, MIME"text/plain"(), cpdag(chain)))
    end
end

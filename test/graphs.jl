@testset "graphs" begin
    @rv A B C D E
    chain = @dag A → B → C
    fork = @dag B → (A, C)
    collider = @dag (A, C) → B

    @test d_separated(chain, A, C, B) && !d_separated(chain, A, C)
    @test d_separated(fork, A, C, B) && !d_separated(fork, A, C)
    @test !d_separated(collider, A, C, B) && d_separated(collider, A, C)

    # conditioning on a descendant of a collider opens it
    g = @dag begin
        (A, C) → B
        B → D
    end
    @test d_separated(g, A, C) && !d_separated(g, A, C, D)

    @test parents(g, B) == [A, C]
    @test Set(children(g, B)) == Set([D])
    @test ancestors(g, D) == Set([A, B, C])
    @test descendants(g, A) == Set([B, D])
    @test markov_blanket(g, A) == Set([B, C])
    order = topological_order(g)
    @test findfirst(==(B), order) < findfirst(==(D), order)
    @test_throws ArgumentError DAG(A => B, B => A)

    # graph surgery
    @test isempty(parents(remove_incoming(g, B), B))
    @test isempty(children(remove_outgoing(g, B), B))

    # independence statements
    @test (A ⟂ C) isa Independence
    @test holds(A ⟂ C, g) && !holds(A ⟂ C | D, g)
    @test holds(A ⟂ C | B, chain)
    ind = independencies(chain)
    @test length(ind) == 1 && ind[1] == (A ⟂ C | B)

    # macro forms
    @test edges(@dag A --> B) == [A => B]
    @test edges(@dag A => B) == [A => B]
    @test Set(nodes(@dag(A → B, E))) == Set([A, B, E])
end

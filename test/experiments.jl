@testset "experiments" begin
    @rv D1 D2 S Car Pick Host C1 C2 F
    @event Doubles BothAces Heads

    dice = experiment(D1 => die(), D2 => die(), S => derived(+, D1, D2), Doubles => derived(==, D1, D2))
    @test compute(P(S(7)), dice) === 1//6
    @test compute(P(Doubles | S(8)), dice) == 1//5
    @test compute(P(Doubles), dice) == 1//6
    @test domain(dice, S) == collect(2:12)
    @test verify(P(D1 | D2), P(D1), dice; atol=0)                 # independent dice
    @test compute(E(S), dice) == 7

    monty = experiment(Car => uniform(1:3), Pick => uniform(1:3),
                       Host => depends_on((c, p) -> uniform(setdiff(1:3, [c, p])), Car, Pick))
    @test compute(P(Car(2) | Pick(1), Host(3)), monty) == 2//3    # switching wins
    @test compute(P(Car(1) | Pick(1), Host(3)), monty) == 1//3

    u = urn(:red => 3, :blue => 2)
    without = experiment(draws(u, C1, C2))
    @test compute(P(C2(:red)), without) == 3//5                   # exchangeable
    @test compute(P(C2(:red) | C1(:red)), without) == 1//2
    @test compute(P(C1(:red) | C2(:blue)), without) == 3//4
    with = experiment(draws(u, C1, C2; replace=true))
    @test compute(P(C2(:red) | C1(:red)), with) == 3//5

    ace(card) = card[1] == :A
    cards = experiment(draws(deck(), C1, C2), BothAces => derived((a, b) -> ace(a) && ace(b), C1, C2))
    @test compute(P(C1((:A, :♠))), cards) == 1//52
    @test compute(P(BothAces), cards) == 1//221

    biased = experiment(F => coin(1//3), Heads => derived(==(:heads), F))
    @test compute(P(Heads), biased) == 1//3
    @test compute(P(F(:tails)), experiment(F => categorical(:heads => 0.25, :tails => 0.75))) ≈ 0.75
    @test compute(P(Heads), experiment(Heads => bernoulli(2//5))) == 2//5

    @test_throws ArgumentError experiment(S => derived(+, D1, D2))             # parents come first
    @test_throws ArgumentError experiment(D1 => die(), D1 => die())
    @test_throws ArgumentError experiment(Doubles => uniform(1:2))              # events are true/false
    @test_throws ArgumentError categorical(:a => 0.5, :b => 0.6)
end

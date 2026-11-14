@testset "odds" begin
    @event H D D2

    @test odds(P(H | D)) == P(H | D) / P(H' | D)
    @test likelihood_ratio(D, H) == P(D | H) / P(D | H')
    @test bayes_factor(D, H, D2) == P(D | H) / P(D | D2)
    @test bayes_odds(P(H | D)) == (P(D | H) / P(D | H')) * (P(H) / P(H'))

    # posterior odds = likelihood ratio × prior odds, exactly
    bn = BayesNet(H => 1//100, (D | H) => [9//10, 1//20])
    post = compute(odds(P(H | D)), bn)
    @test post == compute(bayes_odds(P(H | D)), bn)
    @test post == 2//11
    @test probability_from_odds(post) == 2//13 == compute(P(H | D), bn)
    @test odds_from_probability(1//4) == 1//3

    # on the log scale, conditionally independent evidence adds
    bn2 = BayesNet(H => 1//10, (D | H) => [4//5, 1//10], (D2 | H) => [1//2, 1//4])
    lhs = log_odds(P(H | D, D2))
    rhs = weight_of_evidence(D, H) + weight_of_evidence(D2, H) + log_odds(P(H))
    @test compute(lhs, bn2) ≈ compute(rhs, bn2)

    # two hypotheses against each other
    @rv M
    bn3 = BayesNet(M => [1//2, 1//3, 1//6], (D | M) => [1//2, 1//4, 3//4])
    @test compute(bayes_odds(P(M(1) | D), P(M(2) | D)), bn3) == compute(P(M(1) | D) / P(M(2) | D), bn3)
    @test_throws ArgumentError odds(P(M | D))
end

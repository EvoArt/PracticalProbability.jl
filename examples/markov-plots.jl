using PracticalProbability
import PracticalBayesPlot as PBP

out = isempty(ARGS) ? joinpath(@__DIR__, "markov-output") : only(ARGS)
mkpath(out)
plotbackend!(:pbp)
weather = MarkovChain([3//4 1//4; 1//3 2//3]; states=["Sun", "Rain"])
PBP.save(joinpath(out, "markov.svg"), markovplot(weather))
absorbing = MarkovChain([1 0 0; 1//4 1//2 1//4; 0 0 1]; states=["Lost", "Playing", "Won"])
PBP.save(joinpath(out, "absorbing.svg"), markovplot(absorbing))

@rv X Y
fig = hmmplot(X, Y, 5; observed=[true, true, false, true, true])
PBP.save(joinpath(out, "hmm.svg"), fig)
PBP.save(joinpath(out, "hmm.tex"), fig)
PBP.save(joinpath(out, "hmm.typ"), fig)

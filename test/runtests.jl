using PracticalProbability
using PracticalProbability: Product, Quotient, Marginal, Sum
using Random: Xoshiro
using Test

# The PracticalBayes extension is tested only where PracticalBayes (and
# Distributions, which its models call) can be loaded.
const HAVE_PB = Base.find_package("PracticalBayes") !== nothing &&
                Base.find_package("Distributions") !== nothing
if HAVE_PB
    using PracticalBayes: PracticalBayes
    using Distributions: Normal, Exponential
end

# Every symbolic identity is also checked numerically: an equation that holds
# must compute equal (exactly, by enumeration) on random networks.
const RNG = Xoshiro(20261112)

# Whether any term in an expression still contains a do(⋅).
has_do(e::Prob) = any(a -> a isa Intervention, e.given)
has_do(e::Union{Product,Quotient,Marginal,Sum}) = any(has_do, PracticalProbability.children(e))
has_do(::Literal) = false
has_do(::PExpr) = false

@testset "PracticalProbability" begin
    include("terms.jl")
    include("graphs.jl")
    include("rules.jl")
    include("events.jl")
    include("densities.jl")
    include("numeric.jl")
    include("output.jl")
    include("generative.jl")
    include("assumptions.jl")
    include("expectations.jl")
    include("plates.jl")
    include("importance.jl")
    include("drawing.jl")        # PracticalBayesPlot drawings (before Makie may load)
    include("venn.jl")           # the Makie part only where CairoMakie loads
    include("diagrams.jl")       # likewise
    include("information.jl")
    include("shannon.jl")
    include("identification.jl")
    include("counterfactuals.jl")
    include("causation.jl")
    include("transport.jl")
    if HAVE_PB
        include("practicalbayes.jl")
    else
        @info "PracticalBayes/Distributions not in the test environment; skipping extension tests"
    end
end

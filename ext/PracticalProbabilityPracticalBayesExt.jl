module PracticalProbabilityPracticalBayesExt

# A PracticalBayes `Model` holds the generated evaluator (`m.f`, `_name_eval`),
# its arguments and what it was conditioned on, but not its source. The graph
# comes from the registry `@graphical` fills at definition time; the model
# instance decides what is data: an argument that is not `missing`, or a name
# bound with `condition`/`|`.

using PracticalProbability: PracticalProbability, GenerativeModel, generative, observe, unobserve
using PracticalBayes: PracticalBayes

function PracticalProbability.generative(m::PracticalBayes.Model)
    gm = generative(m.f)
    stoch = Set(v.name for v in vcat(gm.latent, gm.observed))
    obs = Symbol[]
    lat = Symbol[]
    for name in stoch
        if haskey(m.conditioned, name)
            push!(obs, name)
        elseif haskey(m.args, name)
            getfield(m.args, name) === missing ? push!(lat, name) : push!(obs, name)
        else
            push!(lat, name)
        end
    end
    return observe(unobserve(gm, lat...), obs...)
end

end

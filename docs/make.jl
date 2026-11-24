using Documenter, PracticalProbability

makedocs(
    sitename = "PracticalProbability.jl",
    modules = [PracticalProbability],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", "false") == "true",
                             size_threshold = nothing),
    pages = ["Home" => "index.md", "API" => "api.md"],
    checkdocs = :exports,
    warnonly = [:missing_docs, :cross_references],
)

deploydocs(repo = "github.com/EvoArt/PracticalProbability.jl.git", devbranch = "main")

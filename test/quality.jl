# Static checks: Aqua (ambiguities, unbound type parameters, stale or
# uncompat'd dependencies, piracy) and JET (type errors and undefined names).
# Each runs only where its package loads, so the suite still runs without them.

if Base.find_package("Aqua") !== nothing
    @eval using Aqua: Aqua
    @testset "Aqua" begin
        Aqua.test_all(PracticalProbability; ambiguities=false, persistent_tasks=false)
        Aqua.test_ambiguities(PracticalProbability)
    end
else
    @info "Aqua not in the test environment; skipping"
end

if Base.find_package("JET") !== nothing && VERSION >= v"1.11"
    @eval using JET: JET
    @testset "JET" begin
        # errors JET can prove (undefined names, no matching method), in this package's own code
        JET.test_package(PracticalProbability; target_modules=(PracticalProbability,), mode=:basic)
    end
else
    @info "JET not in the test environment (or Julia < 1.11); skipping"
end

using JSON
using Arrow
using Test
using Tables
using Voiage

@testset "EVPI" begin
    @test evpi([10.0 1.0; 2.0 8.0]) == 3.0
    @test evpi(reshape(Float64[], 0, 2)) == 0.0
end

@testset "Rust-normalized contracts" begin
    decision_path = normpath(joinpath(
        @__DIR__, "..", "..", "..", "specs", "core-api", "examples", "v1",
        "decision-problem.example.json",
    ))
    decision = normalize_decision_problem(JSON.parsefile(decision_path))
    @test decision["decision_problem_id"] == "screening-program-001"

    assurance_path = normpath(joinpath(
        @__DIR__, "..", "..", "..", "specs", "core-api", "examples", "v1",
        "statistical-assurance.example.json",
    ))
    assurance = normalize_statistical_assurance(read(assurance_path, String))
    @test assurance["reporting_class"] == "nested-monte-carlo"
end

@testset "Canonical Arrow contract" begin
    path = tempname() * ".arrow"
    Arrow.write(path, (
        reporting_class=["analytic"],
        replications=UInt64[1],
        stopping_reason=["closed-form"],
        has_confidence_interval=[false],
        has_convergence_evidence=[false],
        has_rng_identity=[false],
        payload_json=["{\"reporting_class\":\"analytic\"}"],
    ))
    table = read_voiage_arrow(path)
    @test :payload_json in Tables.columnnames(table)
    rm(path; force=true)
end

@testset "Shared numerical reference" begin
    fixture_path = normpath(joinpath(
        @__DIR__, "..", "..", "..", "specs", "numerical-reference", "v1", "evpi-cases.json"
    ))
    reference = JSON.parsefile(fixture_path)
    @test reference["schema_version"] == "1.0.0"
    @test reference["method"] == "evpi"

    for fixture_case in reference["cases"]
        matrix = reduce(
            vcat,
            (permutedims(Float64.(row)) for row in fixture_case["net_benefits"]),
        )
        expected = Float64(fixture_case["expected"]["value"])
        atol = Float64(fixture_case["expected"]["atol"])
        @test isapprox(evpi(matrix), expected; atol=atol, rtol=0.0)
    end
end

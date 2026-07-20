using JSON
using Test
using Voiage

@testset "EVPI" begin
    @test evpi([10.0 1.0; 2.0 8.0]) == 3.0
    @test evpi(reshape(Float64[], 0, 2)) == 0.0
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

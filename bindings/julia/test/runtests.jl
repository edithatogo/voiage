using JSON
using Test
using Voiage

if get(ENV, "VOIAGE_RUN_JULIA_AQUA", "1") == "1"
    using Aqua
    @testset "Package quality" begin
        Aqua.test_all(Voiage)
    end
end

@testset "EVPI" begin
    @test evpi([10.0 1.0; 2.0 8.0]) == 3.0
    @test evpi(reshape(Float64[], 0, 2)) == 0.0
end

@testset "ENBS" begin
    @test enbs(12.5, 3.0) == 9.5
    @test enbs(2.0, 3.0) == -1.0
    @test_throws ArgumentError enbs(NaN, 3.0)
    @test_throws ArgumentError enbs(12.5, -1.0)
end

@testset "Shared numerical reference" begin
    fixture_path = normpath(joinpath(@__DIR__, "fixtures", "evpi-cases.json"))
    @test isfile(fixture_path)
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

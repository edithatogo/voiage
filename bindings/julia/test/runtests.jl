using Test
using Voiage

@testset "EVPI" begin
    @test evpi([10.0 1.0; 2.0 8.0]) == 3.0
    @test evpi(reshape(Float64[], 0, 2)) == 0.0
end

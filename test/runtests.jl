using GradOptimizer
using Test

@testset "GradOptimizer.jl" begin
    # Write your tests here.

    epochs = 10000
    gd_param = gd_parameter([0.0,0.0], [0.0,0.0], 1.0e-3, epochs, 0.001, true, 100, 0.01, 0.9, 0.999, 1.0e-6, loss_func, 1)

    average_x = 3.0
    sigma_x = 3.0

    x = randn(32^3)
    y = randn(32^3)

    #平均と標準偏差をずらした正規分布
    x = sigma_x * x .+ average_x
    θ = [0.0, 1.0]

    θ, loss = train_commit(x, y, θ, gd_param)

    @test loss < 0.001
    @test 2.5 < θ[1] && θ[1] < 3.5
    @test 2.5 < θ[2] && θ[2] < 3.5

end

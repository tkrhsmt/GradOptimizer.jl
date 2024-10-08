using GradOptimizer
using Test

@testset "GradOptimizer.jl" begin
    # Write your tests here.

    function loss_func(data1, data2, θ)

        data2_1 = θ[2] * data2 .+ θ[1]

        return data1, data2_1
    end

    epochs = 10000
    gd_param = gd_parameter([0.0,0.0], [0.0,0.0], 1.0e-3, epochs, 0.01, true, 100, 0.01, 0.9, 0.999, 1.0e-6, loss_func, 1)

    average_x = 3.0
    sigma_x = 3.0

    x = randn(32^3)
    y = randn(32^3)

    #平均と標準偏差をずらした正規分布
    x = sigma_x * x .+ average_x
    θ = [0.0, 1.0]

    θ, loss = train_commit(x, y, θ, gd_param)

    @test loss < 0.01
    @test 2.5 < θ[1] && θ[1] < 3.5
    @test 2.5 < θ[2] && θ[2] < 3.5
end

@testset "mutual_information" begin

    function loss_func(data1, data2, θ)

        data2_1 = θ[1] * data2 + data1

        return data1, data2_1
    end

    average_x = 3.0
    sigma_x = 3.0

    x = randn(32^3)
    y = randn(32^3)

    θ = [1.0]

    mi_max = GradOptimizer.mutual_info(x, x)
    println("mutual information maximum : $mi_max")

    epochs = 1000
    gd_param = gd_parameter([0.0], [0.0], 1.0e-3, epochs, 0.001-mi_max, true, 100, 0.01, 0.9, 0.999, 1.0e-6, loss_func, 2)

    θ, loss = train_commit(x, y, θ, gd_param)

    println(θ[1])
    @test -0.01 < θ[1] && θ[1] < 0.01

end

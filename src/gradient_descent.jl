export gd_parameter, train_commit

using Statistics
using StatsBase

"""
    mutable struct gd_parameter

A mutable structure that stores parameters for gradient descent optimization using the Adam optimizer.

# Fields
- `ν::Array{Float64}`: First moment estimate (for Adam optimizer).
- `s::Array{Float64}`: Second moment estimate (for Adam optimizer).
- `ϵ::Float64`: Small value to prevent division by zero.
- `epochs::Int`: Number of training epochs.
- `loss_eps::Float64`: Threshold for the loss to stop training.
- `log::Bool`: Whether to log the loss during training.
- `log_num::Int`: Span that outputs logs
- `α::Float64`: Learning rate for Adam optimizer.
- `β1::Float64`: Exponential decay rate for the first moment (for Adam optimizer).
- `β2::Float64`: Exponential decay rate for the second moment (for Adam optimizer).
- `adam_ϵ::Float64`: Small constant added to avoid division by zero in Adam optimizer.
- `loss_function::Function`: Custom loss function for comparing data sets.
- `select_lossfunc::Int`: Determines which loss function to use (1 for KL divergence, other values for mean squared error).
"""
mutable struct gd_parameter
    ν :: Array{Float64}
    s :: Array{Float64}
    ϵ :: Float64

    epochs :: Int
    loss_eps :: Float64

    log :: Bool
    log_num :: Int

    α :: Float64
    β1 :: Float64
    β2 :: Float64
    adam_ϵ :: Float64

    loss_function :: Function
    select_lossfunc :: Int
end

"""
    kl_divergence(data1, data2)

Calculates the Kullback-Leibler (KL) divergence between two datasets using histograms to approximate probability density functions.

# Arguments
- `data1::Vector{Float64}`: First dataset.
- `data2::Vector{Float64}`: Second dataset.

# Returns
- `kl_div::Float64`: The computed KL divergence between the estimated probability distributions of the two datasets.

# Notes
- A histogram is used to estimate the probability density for both datasets.
- The bin width (`dx`) is determined using the Freedman-Diaconis rule.
- To prevent division by zero, a small constant (`1.0e-4`) is added to both probability distributions.
"""
function kl_divergence(data1, data2)

    #推定範囲の設定
    dx = 3.49 * std(data1) / (length(data1)^(1/3))
    x = [minimum(vcat(data1,data2)):dx:maximum(vcat(data1,data2));]

    #ヒストグラムによる確率密度分布の推定
    p = fit(Histogram, data1, x).weights / length(data1)
    q = fit(Histogram, data2, x).weights / length(data2)

    #ゼロ割を防ぐための誤差量を追加
    p .+= 1.0e-4
    q .+= 1.0e-4
    #KL情報量の計算
    kl_div = sum(p .* log.(p ./ q))

    return kl_div
end

function mutual_info(data1, data2)

    #推定範囲の設定
    dx = 3.49 * std(data1) / (length(data1)^(1/3))
    x = [minimum(data1):dx:maximum(data1);]
    dy = 3.49 * std(data2) / (length(data2)^(1/3))
    y = [minimum(data2):dy:maximum(data2);]

    #ヒストグラムによる確率密度分布の推定
    p = fit(Histogram, (data1, data2), (x, y)).weights / length(data1)

    #ゼロ割を防ぐための誤差量を追加
    p1 = sum(p, dims=2)
    p2 = sum(p, dims=1)

    #相互情報量の計算
    num1 = length(p1)
    num2 = length(p2)
    mi = 0.0
    for j in 1:num2
        for i in 1:num1
            if p[i, j] != 0 && p1[i] != 0 && p2[j] != 0
                mi += p[i, j] * (log(p[i, j] / (p1[i] * p2[j])))
            end
        end
    end

    return mi

end

"""
    Loss_function(data1, data2, θ, gd_prm)

Calculates the loss based on the selected loss function (KL divergence or mean squared error).

# Arguments
- `data1::Vector{Float64}`: First dataset.
- `data2::Vector{Float64}`: Second dataset.
- `θ::Vector{Float64}`: Model parameters.
- `gd_prm::gd_parameter`: Gradient descent parameters including the selected loss function.

# Returns
- `loss::Float64`: Computed loss based on the chosen loss function.

# Notes
- gd_prm.select_lossfunc == 1 : KL Divergence
- gd_prm.select_lossfunc == 2 : mutual information
- otherwise : Squared Error
"""
function Loss_function(data1, data2, θ, gd_prm)

    data1_1, data2_1 = gd_prm.loss_function(data1, data2, θ)

    #誤差関数による誤差量の計算
    if gd_prm.select_lossfunc == 1
        # KL情報量
        return kl_divergence(data1_1, data2_1)
    elseif gd_prm.select_lossfunc == 2
        # 相互情報量
        return - mutual_info(data1_1, data2_1)
    else
        # 二乗誤差
        return sum((data1_1 .- data2_1).^ 2)
    end
end

"""
    numerical_gradient(data1, data2, gd_prm, θ)

Computes the numerical gradient of the loss function with respect to model parameters.

# Arguments
- `data1::Vector{Float64}`: First dataset.
- `data2::Vector{Float64}`: Second dataset.
- `gd_prm::gd_parameter`: Gradient descent parameters.
- `θ::Vector{Float64}`: Current model parameters.

# Returns
- `grad::Vector{Float64}`: Computed gradient.
- `loss_now::Float64`: Loss value at the current parameters.

# Notes
- Uses finite differences to estimate the gradient of the loss function.
"""
function numerical_gradient(data1, data2, gd_prm, θ)

    #現在のパラメータにおける誤差量の計算
    loss_now = Loss_function(data1, data2, θ, gd_prm)

    #パラメータを微小に変化させた時における誤差量を計算
    θ_len = length(θ)
    θ_plus = zeros(θ_len, θ_len)
    for j in 1:θ_len
        for i in 1:θ_len
            if i == j
                θ_plus[i, j] = θ[i] + gd_prm.ϵ
            else
                θ_plus[i, j] = θ[i]
            end
        end
    end

    #微小に変更したパラメータにおける誤差量を計算
    loss_Δ = zeros(θ_len)
    for i in θ_len
        loss_Δ[i] = Loss_function(data1, data2, θ_plus[:, i], gd_prm)
    end

    #勾配を計算
    grad = (loss_Δ .- loss_now) / gd_prm.ϵ

    return grad, loss_now

end

"""
    train_step(data1, data2, θ, gd_prm)

Performs a single training step, including gradient computation and parameter update using the Adam optimizer.

# Arguments
- `data1::Vector{Float64}`: First dataset.
- `data2::Vector{Float64}`: Second dataset.
- `θ::Vector{Float64}`: Model parameters.
- `gd_prm::gd_parameter`: Gradient descent parameters.

# Returns
- `θ::Vector{Float64}`: Updated parameters.
- `loss::Float64`: Computed loss.
- `gd_prm::gd_parameter`: Updated gradient descent parameters.
"""
function train_step(data1, data2, θ, gd_prm)

    #現在の勾配を計算
    grad, loss = numerical_gradient(data1, data2, gd_prm, θ)
    #Adam optimizerに従ってパラメータを修正
    θ, gd_prm = adam_optimizer!(θ, grad, gd_prm)

    return θ, loss, gd_prm
end

"""
    train_commit(data1, data2, θ, gd_prm)

Executes the training process for a specified number of epochs, updating parameters at each step.

# Arguments
- `data1::Vector{Float64}`: First dataset.
- `data2::Vector{Float64}`: Second dataset.
- `θ::Vector{Float64}`: Initial model parameters.
- `gd_prm::gd_parameter`: Gradient descent parameters.

# Returns
- `θ::Vector{Float64}`: Final trained parameters.
- `loss::Float64`: Final loss value.

# Notes
- If `gd_prm.log` is `true`, it logs the loss and parameters at each epoch.
"""
function train_commit(data1, data2, θ, gd_prm)

    if gd_prm.log
        println("epoch |\t loss\t θ")
    end

    loss = 0.0
    loss_min = 10000.0
    θ_min = θ

    for epoch in 1:gd_prm.epochs
        θ, loss, gd_prm = train_step(data1, data2, θ, gd_prm)

        if loss < loss_min
            θ_min = θ
            loss_min = loss
        end

        if loss < gd_prm.loss_eps
            if gd_prm.log
                println("train end! :: last epoch -> $epoch :: loss -> $loss_min")
                break
            end
        end

        if gd_prm.log && epoch % gd_prm.log_num == 0
            println("$epoch |\t$loss\t $θ")
        end

    end

    return θ_min, loss_min
end

"""
    adam_optimizer!(θ, grad, gd_prm)

Updates the model parameters using the Adam optimization algorithm.

# Arguments
- `θ::Vector{Float64}`: Model parameters.
- `grad::Vector{Float64}`: Computed gradient.
- `gd_prm::gd_parameter`: Gradient descent parameters including Adam-specific terms.

# Returns
- `θ::Vector{Float64}`: Updated parameters.
- `gd_prm::gd_parameter`: Updated gradient descent parameters.

# Notes
- Applies bias correction to the moment estimates and updates the parameters using the learning rate and moment estimates.
"""
function adam_optimizer!(θ, grad, gd_prm)

    ν2 = gd_prm.β1 * gd_prm.ν .+ (1.0 - gd_prm.β1) * grad
    s2 = gd_prm.β2 * gd_prm.s .+ (1.0 - gd_prm.β2) * grad .* grad
    ν̂ = ν2 / (1.0 - gd_prm.β1)
    ŝ = s2 / (1.0 - gd_prm.β2)

    gd_prm.ν = ν2
    gd_prm.s = s2

    θ = θ .- gd_prm.α ./ (sqrt.(ŝ) .+ gd_prm.adam_ϵ) .* ν̂

    return θ, gd_prm
end

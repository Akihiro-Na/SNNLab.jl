#=
state2lambdaのtest用コード
Maze環境も使用

T = 1000 ms の場合
FT = Float64の時のfor loopの実行結果
0.315458 seconds (1.98 M allocations: 50.407 MiB, 5.43% gc time, 56.93% compilation time)
0.273598 seconds (1.80 M allocations: 38.085 MiB, 51.37% gc time)
0.132554 seconds (1.80 M allocations: 38.085 MiB, 2.04% gc time)
0.132685 seconds (1.80 M allocations: 38.085 MiB, 1.64% gc time)
0.161758 seconds (1.80 M allocations: 38.085 MiB, 18.23% gc time)
0.143962 seconds (1.80 M allocations: 38.085 MiB)

FT = Float32の時のfor loopの実行結果
0.362124 seconds (2.04 M allocations: 49.872 MiB, 2.17% gc time, 54.42% compilation time)
0.329408 seconds (1.80 M allocations: 33.507 MiB, 52.07% gc time)
0.164070 seconds (1.80 M allocations: 33.507 MiB, 1.68% gc time)
0.159756 seconds (1.80 M allocations: 33.507 MiB, 2.40% gc time)
0.160440 seconds (1.80 M allocations: 33.507 MiB, 3.50% gc time)
0.146417 seconds (1.80 M allocations: 33.507 MiB)
=#

using SNNLab
using Plots
using ProgressBars
using Parameters: @unpack # or using UnPack

mutable struct SaveArr{FT,UIT}
    sampling_step::UIT
    saveindmax::UIT
    statearr::Array{FT}
    spikearr::BitArray
    Isynarr::Array{FT}
    idx::UIT

    function SaveArr{FT,UIT}(dt::FT, nt::UIT, Ninput::UIT, env) where {FT, UIT}
        sampling_step = div(10, dt)
        saveindmax = UIT(div(nt, sampling_step) + 1)
        statearr = zeros(FT, saveindmax, length(env.state))
        spikearr = BitArray(undef, nt, Ninput)
        Isynarr = zeros(FT, saveindmax, Ninput)
        idx = 1
        new{FT,UIT}(sampling_step, saveindmax, statearr, spikearr, Isynarr, idx)
    end
end

function run_state2lambda_test()
    FT = Float64
    UIT = UInt32

    T = 1000 # ms
    dt::FT = 0.01 # ms
    nt::UIT = div(T, dt) # number of timesteps
    t = Array{FT}(1:nt) * dt


    # Mazemodelの定義 ========
    env = Maze{FT}(start=(1, 1))

    #init!(env, (1,1),0)
    # =========================

    # state2lambdaparameterの定義 ===
    lambda = State2λ{FT}()
    Ninput = lambda.param.N
    # ===============================

    # PoissonNeuronの定義 ===========
    neurons = PPPNeuron{FT}(N=Ninput, nt=nt)
    synapses = DExpSynapse{FT}(N=Ninput)

    # 記録用配列の確保 ==============
    
    savearr = SaveArr{FT,UIT}(dt, nt, Ninput, env)

    init!(neurons)
    # ===============================



    # simulation
    @time for i = 1:nt
        # input neuron
        update!(lambda, lambda.param, env.state) # 1 allocation
        update!(neurons, dt, lambda.λvec)
        savearr.spikearr[i, :] = neurons.spike # 1 allocation

        # synapse
        update!(synapses, synapses.param, dt, neurons.spike)

        # env
        action::FT = 2 * rand() - 1 # random action # 1 allocation
        update!(env, env.param, action, dt) # 1 allocation

        if mod1(i, savearr.sampling_step) == 1
            savearr.Isynarr[savearr.idx, :] = synapses.Isyn # 1 allocation
            #copyto!(savearr.statearr[savearr.idx, :], env.state)
            savearr.statearr[savearr.idx, :] .= env.state
            savearr.idx += 1
        end
    end

    #アニメーションのインスタンス生成

    function plot_receptive_centers!(receptive_centers::Vector{Tuple{Float64,Float64}}, Isynarr::Matrix{Float64}, timestep::Int)
        x_coords = [center[1] for center in receptive_centers]
        y_coords = [center[2] for center in receptive_centers]

        # 指定したタイムステップの Isyn 値を取得
        Isyn_values = Isynarr[timestep, :]

        # 点の大きさと色を設定
        sizes = abs.(Isyn_values) .* 1000  # サイズのスケーリング（適宜調整）
        colors = Isyn_values  # 色のスケーリング

        scatter!(x_coords, y_coords, size=(600, 600), st=:scatter,
            marker_z=colors, markersize=sizes, c=:viridis, legend=false,
            xlabel="X", ylabel="Y",
            clims=(0, 0.01),
            title="Receptive Centers at timestep $timestep")
    end

    function plot_circle!(x, y, r; color=:red)
        θ = range(0, 2π, 100)
        xc = x .+ r .* cos.(θ)
        yc = y .+ r .* sin.(θ)
        plot!(xc, yc, seriestype=:shape, lw=2, color=color)
    end
    
    anim = Animation()
    xylim = (-2, 22)
    df::UIT = 1
    for i in ProgressBar(1:df:savearr.saveindmax)
        @unpack goal, goal_radius = env.param
        x, y = savearr.statearr[i, :]

        #plot animatin
        plot([x], [y], size=(250, 250), st=:scatter,
            xlims=xylim, ylims=xylim)
        plt = plot_circle!(goal[1], goal[2], goal_radius; color=:red)
        plot_receptive_centers!(lambda.param.receptive_centers, savearr.Isynarr, i)
        frame(anim, plt)
    end
    gif(anim, "maze.gif", fps=50)

end
run_state2lambda_test()
#=
state2lambdaのtest用コード
Maze環境も使用

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

function plot_circle!(x, y, r; color=:red)
    θ = range(0, 2π, 100)
    xc = x .+ r .* cos.(θ)
    yc = y .+ r .* sin.(θ)
    plot!(xc, yc, seriestype=:shape, lw=2, color=color)
end

const FT = Float64

T = 1000 # ms
dt::FT = 0.01 # ms
nt = UInt32(T / dt) # number of timesteps
t = Array{FT}(1:nt) * dt


# Mazemodelの定義 ========
env = Maze{FT}(start=(1, 1))
statearr = zeros(FT, nt, length(env.state))
#init!(env, (1,1),0)
# =========================

# state2lambdaparameterの定義 ===
lambda = state2λ{FT}()
Ninput = lambda.param.N
# ===============================

# PoissonNeuronの定義 ===========
neurons = PPPNeuron{FT}(N=Ninput,nt=nt)
synapses = DExpSynapse{FT}(N=Ninput)
# 記録用
spikearr = BitArray(undef,nt, Ninput)
Isynarr = zeros(FT, nt, Ninput)
init!(neurons)
# ===============================

# simulation
@time for i = 1:nt
    # input neuron
    update!(lambda ,lambda.param , env.state) # 1 allocation
    update!(neurons, dt, lambda.λvec)
    spikearr[i,:] = neurons.spike # 1 allocation
    
    # synapse
    update!(synapses, synapses.param, dt, neurons.spike)
    Isynarr[i,:] = synapses.Isyn # 1 allocation

    # env
    action::FT = 2 * rand() - 1 # random action # 1 allocation
    update!(env,env.param,action,dt) # 1 allocation
    copyto!(statearr[i, :], env.state)
end

#アニメーションのインスタンス生成
#=
anim = Animation()
xylim = (-2, 22)

@time for i = 1:nt
    #plot animatin
    plot(env.state, size=(250, 250), st=:scatter,
        xlims = xylim, ylims = xylim)
    plt = plot_circle!(param.goal[1], param.goal[2], param.goal_radius; color=:red)
    if mod(i,1000) == 0
        frame(anim, plt)
    end
end
#gif(anim, "maze.gif", fps=10)

=#
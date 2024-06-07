#=
Frémaux, N., Sprekeler, H., Gerstner, W., 2013. 
Reinforcement Learning Using a Continuous Time Actor-Critic Framework with Spiking Neurons. 
PLOS Computational Biology 9, e1003024. https://doi.org/10.1371/journal.pcbi.1003024
の再現を目指したファイル
全体的なtest用コード
Maze環境，LIF，PoissonNeuron, state2lambda, DoubleExpSynapseを使用

=#

using SNNLab
using Plots
using ProgressBars
using Parameters: @unpack # or using UnPack
using Printf

mutable struct SaveArr{FT,UIT}
    sampling_step::UIT
    saveindmax::UIT
    statearr::Array{FT}
    spikearr::BitArray
    Isynarr::Array{FT}
    idx::UIT

    function SaveArr{FT,UIT}(dt::FT, nt::UIT, Ninput::UIT, env) where {FT, UIT}
        sampling_step = div(100, dt)
        saveindmax = UIT(div(nt, sampling_step) + 1)
        statearr = zeros(FT, saveindmax, length(env.state))
        spikearr = BitArray(undef, nt, Ninput)
        Isynarr = zeros(FT, saveindmax, Ninput)
        idx = 1
        new{FT,UIT}(sampling_step, saveindmax, statearr, spikearr, Isynarr, idx)
    end
end

#function run_state2lambda_test()
    FT = Float64
    UIT = UInt32

    T = 1000 # ms
    dt::FT = 0.01 # ms
    nt::UIT = div(T, dt) # number of timesteps
    t = Array{FT}(1:nt) * dt

    # Mazemodelの定義 ========
    env = Maze{FT}(start=(8, 8))
    #init!(env, (1,1),0)
    # =========================

    # state2lambdaparameterの定義 ===
    lambda = State2λ{FT,UIT}()
    Ninput::UIT = lambda.param.N
    # ===============================

    # PoissonNeuronの定義 ===========
    input_neurons = PPPNeuron{FT}(N=Ninput, nt=nt)
    input_synapses = DExpSynapse{FT}(N=Ninput)
    # ===============================

    # CriticNeuronの定義 =============
    Ncritic::UIT = 100
    critic_neurons = LIF{FT}(N=Ncritic)
    # 隣接行列
    w_input2critic::Matrix{FT} = rand(Ncritic,Ninput) *3.0
    critic_synapses = DExpSynapse{FT}(N=Ncritic)
    td = TDContinuous{FT,UIT}(N=Ncritic)
    critic_ltp = LTPTrace{FT,UIT}(Npost=Ncritic ,Npre=Ninput)
    # ===============================

    # ActorNeuronの定義 ===============
    Nactor::UIT = 60
    actor_neurons = LIF{FT}(N=Nactor)
    # 隣接行列
    w_input2actor::Matrix{FT} = rand(Nactor,Ninput) *3.0
    actor_synapses = DExpSynapse{FT}(N=Nactor)
    s2a = Spike2action{FT,UIT}(Naction=Nactor)
    actor_ltp = LTPTrace{FT,UIT}(Npost=Nactor ,Npre=Ninput)
    # ===============================

    # 記録用配列の確保 ==============
    savearr_input = SaveArr{FT,UIT}(dt, nt, Ninput, env)
    savearr_actor = SaveArr{FT,UIT}(dt, nt, Nactor, env)
    savearr_critic = SaveArr{FT,UIT}(dt, nt, Ncritic, env)
    # ===============================

    init!(input_neurons)
    iter = ProgressBar(1:nt)#nt
    # simulation
    for i in 1:1 # iter
        # input neuron ================================
        update!(lambda, lambda.param, env.state) # 1 allocation
        update!(input_neurons, dt, lambda.λvec)
        savearr_input.spikearr[i, :] = input_neurons.spike # 1 allocation
        # synapse
        update!(input_synapses, input_synapses.param, dt, input_neurons.spike)
        # ==============================================

        # critic_neurons ================================
        Ie_i2c = w_input2critic * input_synapses.Isyn
        update!(critic_neurons, critic_neurons.param, dt, Ie_i2c)
        savearr_critic.spikearr[i, :] = critic_neurons.spike
        # synapse
        update!(critic_synapses, critic_synapses.param, dt, critic_neurons.spike)
        # TD-error
        update!(td, td.param, dt, critic_neurons.spike, env.reward)
        # LTPTrace
        update!(critic_ltp, critic_ltp.param, dt, critic_synapses.Isyn, critic_neurons.spike)
        # w update
        @time global w_input2critic += td.td_error * critic_ltp.∂V_∂wij
        # ===============================================

        # actor_neurons ================================
        Ie_i2a = w_input2actor * input_synapses.Isyn
        update!(actor_neurons, actor_neurons.param, dt, Ie_i2a)
        savearr_actor.spikearr[i, :] = actor_neurons.spike
        # synapse
        update!(actor_synapses, actor_synapses.param, dt, actor_neurons.spike)
        # LTPTrace
        update!(actor_ltp, actor_ltp.param, dt, actor_synapses.Isyn, actor_neurons.spike)
        # w update
        @time global w_input2actor += td.td_error * actor_ltp.∂V_∂wij
        # ===============================================

        # env ===========================================
        #2 * rand() - 1 # random action # 1 allocation
        update!(s2a,s2a.param,actor_synapses.Isyn)
        update!(env, env.param, s2a.action, dt) # 1 allocation
        # ===============================================

        if mod1(i, savearr_input.sampling_step) == 1
            savearr_input.Isynarr[savearr_input.idx, :] = input_synapses.Isyn # 1 allocation
            savearr_input.statearr[savearr_input.idx, :] .= env.state
            savearr_actor.Isynarr[savearr_input.idx, :] = actor_synapses.Isyn
            savearr_input.idx += 1
        end
        set_description(iter, string(@sprintf("TDerror: %f, state %f %f", td.td_error, env.state[1], env.state[2])))
    end

    #アニメーションのインスタンス生成

    function plot_receptive_centers!(receptive_centers::Vector{Tuple{Float64,Float64}}, Isynarr::Matrix{Float64}, timestep::Int)
        x_coords = [center[1] for center in receptive_centers]
        y_coords = [center[2] for center in receptive_centers]

        # 指定したタイムステップの Isyn 値を取得
        Isyn_values = Isynarr[timestep, :]

        # 点の大きさと色を設定
        sizes = abs.(Isyn_values) .* 1  # サイズのスケーリング（適宜調整）
        colors = Isyn_values  # 色のスケーリング

        scatter!(x_coords, y_coords, size=(600, 600), st=:scatter,
            marker_z=colors, markersize=sizes, c=:viridis, legend=false,
            xlabel="X", ylabel="Y",
            clims=(0, 10.0),
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
    for i in ProgressBar(1:df:savearr_input.saveindmax)
        @unpack goal, goal_radius = env.param
        x, y = savearr_input.statearr[i, :]

        #plot animatin
        plot([x], [y], size=(250, 250), st=:scatter,
            xlims=xylim, ylims=xylim)
        plt = plot_circle!(goal[1], goal[2], goal_radius; color=:red)
        plot_receptive_centers!(lambda.param.receptive_centers, savearr_input.Isynarr, i)
        frame(anim, plt)
    end
    gif(anim, "maze.gif", fps=50)

#end
#run_state2lambda_test()
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
using LinearAlgebra # for lmul! and mul!

let 


@kwdef mutable struct SaveArr{FT,UIT}
    Nactor::UIT
    Ninput::UIT
    dt::FT
    nt::FT
    sampling_interval::UIT # ms
    sampling_step::UIT = div(sampling_interval, dt) # [step]
    saveindmax::UIT = UIT(div(nt, sampling_step) + 1)
    # save array
    statearr::Array{FT} = zeros(FT, saveindmax, 2) # 2 is state length
    inputIsynarr::Array{FT} = zeros(FT, saveindmax, Ninput)
    actorIsynarr::Array{FT} = zeros(FT, saveindmax, Nactor)
    wi2c_mean::Array{FT} = zeros(FT, saveindmax, Ninput)
    wi2c_mean_tmp::Array{FT} = zeros(FT, 1, Ninput) # tmp memory
    wi2a_mean::Array{FT} = zeros(FT, saveindmax, 2, Ninput)
    wi2a_mean_tmp::Array{FT} = zeros(FT, 2, Ninput) # tmp memory
    tdarr::Vector{FT} = zeros(FT, saveindmax)
    idx::UIT = 1
    # spikearr = BitArray(undef, nt, Ninput)
end

function run_nicolas2013_test()
    FT = Float32
    UIT = UInt32

    T::FT = 10 * 10^3 # ms
    dt::FT = 1 # ms
    sampling_interval::UIT = 1000# ms for save data interval
    nt::UIT = div(T, dt) # number of timesteps
    t = Array{FT}(1:nt) * dt

    # Maze modelの定義 ========
    env = Maze{FT}(start=[8, 8])
    #init!(env, (1,1),0)
    # =========================

    # SNNAgent modelの定義 ========
    Ncritic::UIT = 100
    Nactor::UIT = 180
    agent = TDLTPAgent{FT,UIT}(Ncritic=Ncritic, Nactor=Nactor, nt=nt)
    init!(agent.network)
    Ninput = agent.lambda.param.N
    #init!(agent, )
    # =========================

    savearr = SaveArr{FT,UIT}(Ninput=Ninput, Nactor=Nactor, nt=nt, dt=dt, sampling_interval=sampling_interval)

    iter = ProgressBar(1:nt)#nt
    # simulation
    @time for i in iter
        # agent =========================================
        update!(agent, dt, env.state, env.reward) # 1 allocation
        # env ===========================================
        update!(env, env.param, agent.s2a.action, dt) # 1 allocation
        # ===============================================
        #println(sum(abs.(agent.network.critic_ltp.∂V_∂wij)))
        #println(agent.network.td.td_error)
        # save data =====================================
        if mod1(i, savearr.sampling_step) == 1
            idx = savearr.idx
            savearr.statearr[idx, :] = env.state
            savearr.inputIsynarr[idx, :] = agent.network.input_synapses.Isyn
            savearr.actorIsynarr[idx, :] = agent.network.actor_synapses.Isyn
            #savearr.wi2c_mean[savearr.idx, :] = ones(Ncritic)' * agent.network.w_input2critic / Ncritic
            
            sum!(savearr.wi2c_mean_tmp, agent.network.w_input2critic)
            lmul!(1 / Ncritic, savearr.wi2c_mean_tmp)
            savearr.wi2c_mean[idx, :] = savearr.wi2c_mean_tmp

            #savearr.wi2a_mean[idx, :, :] = agent.s2a.param.actionset * agent.network.w_input2actor / Nactor 
            mul!(savearr.wi2a_mean_tmp, agent.s2a.param.actionset, agent.network.w_input2actor)
            lmul!(1 / Nactor, savearr.wi2a_mean_tmp)
            savearr.wi2a_mean[idx, :, :] = savearr.wi2a_mean_tmp
            savearr.tdarr[idx] = agent.network.td.td_error
            savearr.idx += 1
        end
        # ===============================================
        set_description(iter, string(@sprintf("action: (%+4f, %+4f), state (%+4f %+4f), reward %+4f",
            agent.s2a.action[1], agent.s2a.action[2], env.state[1], env.state[2], env.reward)))
    end

    #アニメーションのインスタンス生成
    anim = Animation()

    xylim = (-2, 22)
    df::UIT = 1

    sizefactor = 8
    length_receptive_x = length(agent.lambda.param.receptive_x)
    length_receptive_y = length(agent.lambda.param.receptive_y)
    x_heat = collect(LinRange{FT}(agent.lambda.param.xmin, agent.lambda.param.xmax, length_receptive_x * sizefactor))
    y_heat = collect(LinRange{FT}(agent.lambda.param.xmin, agent.lambda.param.xmax, length_receptive_x * sizefactor))
    clim = (0, 3)
    for i in ProgressBar(1:df:savearr.saveindmax)
        @unpack goal, goal_radius = env.param
        x = savearr.statearr[i, 1]
        y = savearr.statearr[i, 2]

        _, _, c_value = SNNLab.Interp2D(reshape(savearr.wi2c_mean[i, :], (length_receptive_y, length_receptive_x))', sizefactor)

        #plot animatin
        plt = heatmap(x_heat, y_heat, c_value, clims=clim)
        SNNLab.plot_circle!(goal[1], goal[2], goal_radius; color=:red)
        X = [i for i in agent.lambda.param.receptive_x, j in 1:length_receptive_y]
        Y = [j for i in 1:length_receptive_x, j in agent.lambda.param.receptive_y]
        dx = reshape(savearr.wi2a_mean[i, 1, :], (length_receptive_y, length_receptive_x)) * 10
        dy = reshape(savearr.wi2a_mean[i, 2, :], (length_receptive_y, length_receptive_x)) * 10
        # プロット
        SNNLab.arrow0!.(X, Y, dx, dy; as=0.4, lw=3, lc=:red, la=1) # 要検証 ##############
        title_str = @sprintf("Timestep %d, td error = %.7f", i, savearr.tdarr[i])
        plot!([x], [y], st=:scatter,
            xlims=xylim, ylims=xylim,
            size=(600, 600), aspect_ratio=1, legend=false,
            xlabel="X", ylabel="Y",
            title=title_str)
        frame(anim, plt)
    end
    gif(anim, "maze.gif", fps=10)

end

run_nicolas2013_test()

end
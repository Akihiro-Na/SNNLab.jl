#=
Poissonニューロンのtest用コード
=#
using SNNLab
using Random
using Plots
using BenchmarkTools

function run_PPPNeuron_test()
    FT = Float64

    T = 1000 # ms
    dt::FT = 0.01 # ms
    nt = UInt32(T / dt) # number of timesteps

    N = UInt32(20) # ニューロンの数

    t = Array{FT}(1:nt) * dt

    # λarrを生成
    phases = 2 * π .* rand(N)
    λarr = 30 .* (sin.(1e-2 .* t' .+ phases)) .^ 2



    # 記録用
    spikearr = BitArray(undef, nt, N)
    Isynarr = zeros(FT, nt, N)

    # modelの定義
    neurons = PPPNeuron{FT}(N=N, nt=nt)
    synapses = DExpSynapse{FT}(N=N)


    # simulation
    init!(neurons)
    for i = 1:nt
        update!(neurons, dt, λarr[:, i])#neurons.random_numbers[i,:] .< λarr[:,i]*dt*1e-3
        spikearr[i, :] = neurons.spike

        # synapse
        update!(synapses, synapses.param, dt, neurons.spike)
        Isynarr[i, :] = synapses.Isyn
    end

    # Plots

    i = 1
    p1 = plot(t, Isynarr[:, i], color="black", legend=false)
    ylabel!(p1, "Synapse current [mA]")
    p2 = plot(t, spikearr[:, i], color="black", legend=false)
    ylabel!(p2, "Spike")
    p3 = plot(t, λarr[i, :], color="black", legend=false)
    ylabel!(p3, "λ  [Hz]")
    xlabel!(p3, "Time [ms]")
    # プロットを縦に2つ並べる
    p = plot(p1, p2, p3, layout=(3, 1), size=(1000, 800))
end

@benchmark run_PPPNeuron_test()
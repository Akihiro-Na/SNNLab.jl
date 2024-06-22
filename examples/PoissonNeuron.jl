#=
Poissonニューロンのtest用コード

テストパラメータ
T = 100*10^3 # ms
dt::FT = 1 # ms
速度テスト old 最初に対応する長さ分ntの乱数を取得
0.008281 seconds (10.55 k allocations: 848.461 KiB, 97.09% compilation time)
0.022971 seconds (100.00 k allocations: 21.362 MiB)
0.022608 seconds (100.00 k allocations: 21.362 MiB)
0.022973 seconds (100.00 k allocations: 21.362 MiB)

速度テスト new 固定長nt_randを取得し使い切るたびに更新 条件判定のためif文も追加
0.050411 seconds (109.60 k allocations: 35.712 MiB, 29.43% gc time, 19.74% compilation time)
0.032647 seconds (100.05 k allocations: 35.097 MiB, 10.23% gc time)
0.026472 seconds (100.05 k allocations: 35.097 MiB, 5.28% gc time)
0.024619 seconds (100.05 k allocations: 35.097 MiB)
0.024829 seconds (100.05 k allocations: 35.097 MiB)
0.032825 seconds (100.05 k allocations: 35.097 MiB)
=#
using SNNLab
using Random
using Plots

function run_PPPNeuron_test()
    FT = Float64
    UIT = UInt32
    T = 1*10^3 # ms
    dt::FT = 1 # ms
    nt = UInt32(T / dt) # number of timesteps
    nt_rand = UInt32(nt / 10)
    N = UInt32(20) # ニューロンの数

    t = Array{FT}(1:nt) * dt

    # λarrを生成
    phases = 2 * π .* rand(N)
    λarr = 30 .* (sin.(1e-2 .* t' .+ phases)) .^ 2



    # 記録用
    spikearr = BitArray(undef, nt, N)
    Isynarr = zeros(FT, nt, N)

    # modelの定義
    neurons = PPPNeuron{FT,UIT}(N=N, nt=nt_rand)
    synapses = DExpSynapse{FT,UIT}(N=N)


    # simulation
    init!(neurons)
    @time for i = 1:nt
        update!(neurons, dt, @view λarr[:, i])#neurons.random_numbers[i,:] .< λarr[:,i]*dt*1e-3
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

run_PPPNeuron_test()
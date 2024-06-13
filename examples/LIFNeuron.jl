#=
LIFニューロンのtest
入力電流に対する応答の確認
=#
using SNNLab
using Plots

function run_LIFNeuron_test()
    FT = Float64

    T = 450 # ms
    dt::FT = 1 # ms
    nt = UInt32(T / dt) # number of timesteps
    N = 3 # ニューロンの数

    # 入力刺激
    t = Array{FT}(1:nt) * dt
    Ie::Matrix{FT} = repeat(25.0 * ((t .> 50) - (t .> 200)) + 50.0 * ((t .> 250) - (t .> 400)), 1, N)  # injection current

    # 記録用
    varr = zeros(FT, nt, N)

    # modelの定義
    neurons = LIF{FT}(N=N)

    # simulation
    @time for i = 1:nt
        update!(neurons, neurons.param, dt, Ie[i, :])
        varr[i, :] = neurons.v_
    end

    # Plots
    i = 1
    p1 = plot(t, varr[:, i], color="black", legend=false)
    ylabel!(p1, "Membrane potential [mV]")
    p2 = plot(t, Ie[:, i], color="black", legend=false)
    ylabel!(p2, "Input current [mA]")
    xlabel!(p2, "Time [s]")
    # プロットを縦に2つ並べる
    p = plot(p1, p2, layout=(2, 1), size=(800, 600))
    display(p)

end

run_LIFNeuron_test()
#=
LIFニューロンのtest
入力電流に対する応答の確認
    T = 450 # ms
    dt::FT = 1 # ms
update!関数 N=4000
  0.026772 seconds
  0.025039 seconds
  0.024609 seconds
  0.024657 seconds
  0.029092 seconds
update_threads!関数 N=4000
  0.025752 seconds (13.95 k allocations: 1.730 MiB)
  0.029489 seconds (13.95 k allocations: 1.730 MiB)
  0.030467 seconds (13.95 k allocations: 1.730 MiB)
  0.029742 seconds (13.95 k allocations: 1.730 MiB)
  0.029824 seconds (13.95 k allocations: 1.730 MiB)
update!関数 N=60 
  0.000178 seconds
  0.000179 seconds
  0.000179 seconds
  0.000178 seconds
  0.000191 seconds
update_threads!関数 N=60
  0.001940 seconds (13.95 k allocations: 1.730 MiB)
  0.002048 seconds (13.95 k allocations: 1.730 MiB)
  0.002030 seconds (13.95 k allocations: 1.730 MiB)
  0.002057 seconds (13.95 k allocations: 1.730 MiB)
  0.002820 seconds (13.95 k allocations: 1.731 MiB)

=#
using SNNLab
using Plots

function run_DExpSynapse_test()
    FT = Float32
    UIT = UInt32
    T = 450 # ms
    dt::FT = 1 # ms
    nt = UInt32(T / dt) # number of timesteps
    N::UIT = 60 # ニューロンの数

    # 入力刺激
    t = Array{FT}(1:nt) * dt
    Ie::Matrix{FT} = repeat(25.0 * ((t .> 50) - (t .> 75)) +
                25.0 * ((t .> 200) - (t .> 250)) +
                50.0 * ((t .> 350) - (t .> 400)), 1, N)  # injection current

    # 記録用
    varr = zeros(FT, nt, N)
    spikearr = zeros(Bool, nt, N)
    Isynarr = zeros(FT, nt, N)

    # modelの定義
    neurons = LIF{FT,UIT}(N=N)
    synapses = DExpSynapse{FT,UIT}(N=N)

    # simulation
    @time for i = 1:nt
        update!(neurons, neurons.param, dt, @view Ie[i, :])
        #update_threads!(neurons, neurons.param, dt, @view Ie[i, :])
        varr[i, :] = neurons.v_
        spikearr[i, :] = neurons.spike

        # synapse
        update!(synapses, synapses.param, dt, neurons.spike)
        #update_threads!(synapses, synapses.param, dt, neurons.spike)
        Isynarr[i, :] = synapses.Isyn
    end

    # Plots

    i = 1
    p1 = plot(t, Isynarr[:, i], color="black", legend=false)
    ylabel!(p1, "Synapse current \n [mA]")
    p2 = plot(t, varr[:, i], color="black", legend=false)
    ylabel!(p2, "Membrane potential \n [mV]")
    p3 = plot(t, Ie[:, i], color="black", legend=false)
    ylabel!(p3, "Input current \n [mA]")
    xlabel!(p3, "Time [s]")
    # プロットを縦に2つ並べる
    p = plot(p1, p2, p3, layout=(3, 1), size=(1000, 800))
end

#@time 
#@code_warntype
run_DExpSynapse_test()
#=
LIFニューロンのtest
入力電流に対する応答の確認
LIF単体でテスト　@time
update_threads! function (並列計算あり)4000ニューロン T=450 ms, dt=1 ms　6スレッド 
  0.006103 seconds (13.95 k allocations: 1.895 MiB)
  0.006208 seconds (13.95 k allocations: 1.895 MiB)
  0.007183 seconds (13.95 k allocations: 1.895 MiB)
  0.007475 seconds (13.95 k allocations: 1.895 MiB)
  0.006522 seconds (13.95 k allocations: 1.895 MiB)
  0.006004 seconds (13.95 k allocations: 1.895 MiB)
update! function (並列計算なし)4000ニューロン T=450 ms, dt=1 ms　6スレッド 
  0.010111 seconds
  0.010299 seconds
  0.010152 seconds
  0.010558 seconds
  0.010624 seconds
  0.010168 seconds
update_threads! function (並列計算あり)60ニューロン T=450 ms, dt=1 ms　6スレッド 
  0.002484 seconds (13.95 k allocations: 1.895 MiB)
  0.002617 seconds (13.95 k allocations: 1.895 MiB)
  0.002238 seconds (13.95 k allocations: 1.895 MiB)
  0.002270 seconds (13.95 k allocations: 1.895 MiB)
update! function (並列計算なし)60ニューロン T=450 ms, dt=1 ms　6スレッド 
  0.000079 seconds
  0.000074 seconds
  0.000076 seconds
  0.000077 seconds
=#
using SNNLab
using Plots

function run_LIFNeuron_test()
    FT = Float32
    UIT = UInt32
    T::FT = 450 # ms
    dt::FT = 1 # ms
    nt = UIT(T / dt) # number of timesteps
    N::UIT = 60 # ニューロンの数

    # 入力刺激
    t = Array{FT}(1:nt) * dt
    Ie::Matrix{FT} = repeat(25.0 * ((t .> 50) - (t .> 200)) + 50.0 * ((t .> 250) - (t .> 400)), 1, N)  # injection current

    # 記録用
    varr = zeros(FT, nt, N)

    # modelの定義
    neurons = LIF{FT,UIT}(N=N)

    # simulation
    @time for i = 1:nt
        update!(neurons, neurons.param, dt, @view Ie[i, :])
        #update_threads!(neurons, neurons.param, dt, @view Ie[i, :])
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

#@time 
#@code_warntype
run_LIFNeuron_test()
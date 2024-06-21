#=
LIFニューロンのtest
入力電流に対する応答の確認
LIF単体でテスト 40000ニューロン T=450 ms, dt=1 ms　6スレッド 
　0.184762 seconds (14.85 k allocations: 70.498 MiB)
  0.177827 seconds (14.85 k allocations: 70.498 MiB, 6.11% gc time)
  0.175320 seconds (14.85 k allocations: 70.498 MiB)
  0.218928 seconds (14.85 k allocations: 70.498 MiB, 27.26% gc time)
  0.163952 seconds (14.85 k allocations: 70.498 MiB)
updateのアロケーションをなくした後
  0.093238 seconds (13.95 k allocations: 1.895 MiB)
  0.089075 seconds (13.95 k allocations: 1.895 MiB)
  0.089431 seconds (13.95 k allocations: 1.895 MiB)
  0.088505 seconds (13.95 k allocations: 1.895 MiB)
  0.091089 seconds (13.95 k allocations: 1.895 MiB)
並列計算なし　40000ニューロン
  0.204907 seconds (900 allocations: 68.685 MiB)
  0.201790 seconds (900 allocations: 68.685 MiB)
  0.210969 seconds (900 allocations: 68.685 MiB, 5.90% gc time)
  0.204427 seconds (900 allocations: 68.685 MiB)
  0.300499 seconds (900 allocations: 68.685 MiB, 26.06% gc time)
updateのアロケーションをなくした後
  0.210940 seconds
  0.198682 seconds
  0.198883 seconds
  0.198432 seconds
  0.208331 seconds
  0.209712 seconds
=#
using SNNLab
using Plots

function run_LIFNeuron_test()
    FT = Float32
    UIT = UInt32
    T::FT = 450 # ms
    dt::FT = 1 # ms
    nt = UIT(T / dt) # number of timesteps
    N::UIT = 40 # ニューロンの数

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
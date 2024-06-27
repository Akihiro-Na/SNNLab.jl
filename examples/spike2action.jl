#=
spike2actionのtest用コード
Poissonニューロンも使用

T = 1000 ms の場合
FT = Float64の時のfor loopの実行結果

FT = Float32の時のfor loopの実行結果

=#

using SNNLab
using Random
using Plots
using ProgressBars

function run_spike2action_test()
    FT = Float32
    UIT = UInt32
    T::FT = 2.5 * 10^3 # ms
    dt::FT = 1 # ms
    nt = UIT(T / dt) # number of timesteps
    nt_rand = UIT(nt / 10)
    N = UIT(20) # ニューロンの数

    t = Array{FT}(1:nt) * dt
    s2a = Spike2action{FT,UIT}(Naction=N)
    # λarrを生成
    function generate_time_series(N, total_time, target_duration, FT::Type)
        time_series = zeros(FT, N, total_time)
        for i in 1:N
            start_time = 100 * (i - 1) % (total_time - target_duration + 1) + 1
            time_series[i, start_time:start_time+target_duration-1] .= one(FT) * 30
        end
        return time_series
    end

    target_duration = 300 # [ms]
    λarr = generate_time_series(N, nt, target_duration, FT)

    # 記録用
    Isynarr = zeros(FT, nt, N)
    actionarr = zeros(FT, nt, 2)
    # modelの定義
    neurons = PPPNeuron{FT,UIT}(N=N, nt=nt_rand)
    synapses = DExpSynapse{FT,UIT}(N=N)


    # simulation
    init!(neurons)
    @time for i = 1:nt
        update!(neurons, dt, @view λarr[:, i])#neurons.random_numbers[i,:] .< λarr[:,i]*dt*1e-3

        # synapse
        update!(synapses, synapses.param, dt, neurons.spike)
        Isynarr[i, :] = synapses.Isyn
        # spike to action
        update!(s2a, s2a.param, synapses.Isyn)
        actionarr[i, :] = s2a.action
    end

    figsize = (1300, 400)
    l = @layout [a{0.3w} b{0.1w} c{0.3w} d{0.3w}]


    # x軸とy軸の範囲を定義
    x_range = 1
    y_range = 1:N
    # グリッドの座標を生成
    grid_coordinates::Vector{Tuple{FT,FT}} = collect(Iterators.product(x_range, y_range))

    #アニメーションのインスタンス生成
    anim = Animation()
    df::UIT = 10
    imax = size(Isynarr, 1)
    for i in ProgressBar(1:df:imax)
        p1 = heatmap(λarr[:, 1:i], colorbar=false)
        xlims!(p1, (0, imax))
        p3 = heatmap(Isynarr[1:i, :]', colorbar=false, clim=(0, 2))
        xlims!(p3, (0, imax))
        p2 = SNNLab.plot_receptive_centers(grid_coordinates, Isynarr, UIT(i))
        p4 = SNNLab.plot_unit_circle()
        p4 = quiver!([0], [0], quiver=([actionarr[i, 1]], [actionarr[i, 2]]), label="")
        plt = plot(p1, p2, p3, p4, layout=l, size=figsize)
        frame(anim, plt)
    end
    #mp4(anim, "spike2action.mp4"; fps=20, loop=0, verbose=false, show_msg=true)
    gif(anim, "spike2action.gif", fps=100)
end

run_spike2action_test()
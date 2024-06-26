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
        start_time = 100*(i - 1) % (total_time - target_duration + 1) + 1
        time_series[i, start_time:start_time+target_duration-1] .= one(FT)*30
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
    update!(s2a,s2a.param,synapses.Isyn)
    actionarr[i, :] = s2a.action
end

function plot_unit_circle_with_vectors()
    # Plot unit circle
    θ = range(0, stop=2π, length=100)
    x = cos.(θ)
    y = sin.(θ)
    
    plot(x, y, label="Unit Circle", aspect_ratio=:equal, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
    scatter!([0], [0], color=:black, label="Origin")
end

size = (900, 400)
l = @layout [a{0.3w} b{0.1w} c{0.3w} d{0.3w}]
p1 = heatmap(λarr,colorbar=false)

# x軸とy軸の範囲を定義
x_range = 1
y_range = 1:N
# グリッドの座標を生成
grid_coordinates::Vector{Tuple{FT,FT}} = collect(Iterators.product(x_range, y_range))

p2 = SNNLab.plot_receptive_centers(grid_coordinates, Isynarr, UIT(10))
p3 = heatmap(Isynarr',colorbar=false)
p4 = plot_unit_circle_with_vectors()
p4 = quiver!([0], [0], quiver=(actionarr[10, :]), label="")
plot(p1, p2, p3, p4, layout = l, size=size)

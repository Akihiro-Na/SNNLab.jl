#=
LIFニューロンのtest
入力電流に対する応答の確認
=#
using SNNLab

T = 450 # ms
dt = 0.01f0 # ms
nt = UInt32(T / dt) # number of timesteps
N = 3 # ニューロンの数

# 入力刺激
t = Array{Float32}(1:nt) * dt
Ie = repeat(25.0f0 * ((t .> 50) - (t .> 200)) + 50.0f0 * ((t .> 250) - (t .> 400)), 1, N)  # injection current

# 記録用
varr = zeros(Float32, nt, N)

# modelの定義
neurons = LIF{Float32}(N=N)

# simulation
@time for i = 1:nt
    update!(neurons, neurons.param, dt, Ie[i, :])
    varr[i, :] = neurons.v_
end

# Plots

i=1
p1 = plot(t, varr[:,i], color="black", legend=false);
ylabel!(p1, "Membrane potential [mV]");
p2 = plot(t, Ie[:,i], color="black", legend=false);
ylabel!(p2, "Input current [mA]");
xlabel!(p2, "Time [s]")
# プロットを縦に2つ並べる
p = plot(p1, p2, layout=(2, 1), size=(800, 600))
display(p)


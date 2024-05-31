#=
LIFニューロンのtest
入力電流に対する応答の確認
=#
using SNNLab
using Plots

T = 450 # ms
dt = 0.01f0 # ms
nt = UInt32(T / dt) # number of timesteps
N = 2 # ニューロンの数

# 入力刺激
t = Array{Float32}(1:nt) * dt
Ie = repeat(25.0f0 * ((t .> 50) - (t .> 75)) + 
            25.0f0 * ((t .> 200) - (t .> 250)) +
            50.0f0 * ((t .> 350) - (t .> 400))
            , 1, N)  # injection current

# 記録用
varr = zeros(Float32, nt, N)
spikearr = zeros(Bool, nt, N)
Isynarr = zeros(Float32, nt, N)

# modelの定義
neurons = LIF{Float32}(N=N)
synapses = DExpSynapse{Float32}(N=N)

# simulation
@time for i = 1:nt
    update!(neurons, neurons.param, dt, Ie[i, :])
    varr[i, :] = neurons.v_
    spikearr[i,:] = neurons.spike
    
    # synapse
    update!(synapses, synapses.param, dt, neurons.spike)
    Isynarr[i,:] = synapses.Isyn
end

# Plots

i=1
p1 = plot(t, Isynarr[:,i], color="black", legend=false);
ylabel!(p1, "Synapse current \n [mA]");
p2 = plot(t, varr[:,i], color="black", legend=false);
ylabel!(p2, "Membrane potential \n [mV]");
p3 = plot(t, Ie[:,i], color="black", legend=false);
ylabel!(p3, "Input current \n [mA]");
xlabel!(p3, "Time [s]");
# プロットを縦に2つ並べる
p = plot(p1, p2, p3, layout=(3, 1), size=(1000, 800))

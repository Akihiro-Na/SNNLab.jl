#=
Maze環境のtest用コード
=#
using SNNLab
using Plots

T = 1000 # ms
dt = 0.01f0 # ms
nt = UInt32(T / dt) # number of timesteps

t = Array{Float32}(1:nt) * dt


# modelの定義
env = Maze()
synapses = DExpSynapse{Float32}(N=N)


# simulation
init!(neurons)
@time for i = 1:nt
    update!(neurons, dt, λarr[:,i])#neurons.random_numbers[i,:] .< λarr[:,i]*dt*1e-3
    spikearr[i,:] = neurons.spike
    
    # synapse
    update!(synapses, synapses.param, dt, neurons.spike)
    Isynarr[i,:] = synapses.Isyn
end

# Plots

i=1
p1 = plot(t, Isynarr[:,i], color="black", legend=false);
ylabel!(p1, "Synapse current [mA]");
p2 = plot(t, spikearr[:,i], color="black", legend=false);
ylabel!(p2, "Spike");
p3 = plot(t, λarr[i,:], color="black", legend=false);
ylabel!(p3, "λ  [Hz]");
xlabel!(p3, "Time [ms]");
# プロットを縦に2つ並べる
p = plot(p1, p2, p3, layout=(3, 1), size=(1000, 800))

#=
Double exponential synapseを定義するファイル
=#

# DoubleExpSynapseのパラメータ(固定)
@kwdef struct DExpSynapseParameter{FT} <: AbstractSynapseParam{FT}
    τ_syn_fast::FT = 5 # 早い時定数 [ms]
    τ_syn_slow::FT = 20 # 遅い時定数(膜の時定数と同じ？) [ms]
    ε0::FT = 20 # [mV・ms] scaling constant
end

# DoubleExpSynapseの定義
@kwdef mutable struct DExpSynapse{FT,UIT} <: AbstractSynapse{FT}
    param::DExpSynapseParameter = DExpSynapseParameter{FT}()
    N::UIT #シナプスの数
    Isyn::Vector{FT} = zeros(N) # シナプス動態
    h::Vector{FT} = zeros(N) # シナプス動態の補助変数
end
#=
# DoubleExpSynapseに対するupdate!メソッドの定義
function update!(synapses::DExpSynapse{FT,UIT}, param::DExpSynapseParameter{FT}, dt::FT, spikes::Vector{FT}) where {FT,UIT}
    @unpack N, Isyn, h = synapses
    @unpack τ_syn_fast, τ_syn_slow, ε0 = param
    
    @inbounds for i = 1:N
        Isyn[i] += dt * (-Isyn[i]/τ_syn_slow + h[i])
        h[i] += dt * (-h[i]/τ_syn_fast + (ε0/dt)*spikes[i]/(τ_syn_fast*τ_syn_slow))
    end
end
=#
# DoubleExpSynapseに対するupdate!メソッドの定義 spikesがBitVector型
function update!(synapses::DExpSynapse{FT,UIT}, param::DExpSynapseParameter{FT}, dt::FT,  spikes::BitVector) where {FT,UIT}
    @unpack N, Isyn, h = synapses
    @unpack τ_syn_fast, τ_syn_slow, ε0= param
    
    @inbounds for i = 1:N
        Isyn[i] += dt * (-Isyn[i]/τ_syn_slow + h[i])
        h[i] += dt * (-h[i]/τ_syn_fast + (ε0/dt)*spikes[i]/(τ_syn_fast*τ_syn_slow))
        # spike による入力電流はdtを変えても変化しないにdtで割っている
    end
end

# DoubleExpSynapseに対するupdate!メソッドの定義 spikesがBitVector型
function update_threads!(synapses::DExpSynapse{FT,UIT}, param::DExpSynapseParameter{FT}, dt::FT,  spikes::BitVector) where {FT,UIT}
    @unpack N, Isyn, h = synapses
    @unpack τ_syn_fast, τ_syn_slow, ε0= param
    
    @inbounds Threads.@threads for i = 1:N
        Isyn[i] += dt * (-Isyn[i]/τ_syn_slow + h[i])
        h[i] += dt * (-h[i]/τ_syn_fast + (ε0/dt)*spikes[i]/(τ_syn_fast*τ_syn_slow))
        # spike による入力電流はdtを変えても変化しないにdtで割っている
    end
end

# DoubleExpSynapseに対するinit!メソッドの定義
function init!(synapses::DExpSynapse)
    @unpack N, Isyn, h = synapses
    Isyn = zeros(N)
    h = zeros(N)
end

function get_Isyn(synapse::DExpSynapse{FT})::Vector{FT} where FT
    return synapse.Isyn
end

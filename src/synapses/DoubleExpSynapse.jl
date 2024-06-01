
using Base: @kwdef
using Parameters: @unpack # or using UnPack

# DoubleExpSynapseのパラメータ(固定)
@kwdef struct DExpSynapseParameter{FT} <: AbstractSynapseParam{FT}
    τ_syn_fast::FT = 5 # 早い時定数 [ms]
    τ_syn_slow::FT = 20 # 遅い時定数(膜の時定数と同じ？) [ms]
end

# DoubleExpSynapseの定義
@kwdef mutable struct DExpSynapse{FT} <: AbstractSynapse{FT}
    param::DExpSynapseParameter = DExpSynapseParameter{FT}()
    N::UInt32 #シナプスの数
    Isyn::Vector{FT} = zeros(N) # シナプス動態
    h::Vector{FT} = zeros(N) # シナプス動態の補助変数
end

# DoubleExpSynapseに対するupdate!メソッドの定義
function update!(synapses::DExpSynapse, param::DExpSynapseParameter, dt, spikes::Vector)
    @unpack N, Isyn, h = synapses
    @unpack τ_syn_fast, τ_syn_slow = param
    
    @inbounds for i = 1:N
        Isyn[i] += dt * (-Isyn[i]/τ_syn_slow + h[i])
        h[i] += dt * (-h[i]/τ_syn_fast + spikes[i]/(τ_syn_fast*τ_syn_slow))
    end
end

function update!(synapses::DExpSynapse, param::DExpSynapseParameter, dt, spikes::BitVector)
    @unpack N, Isyn, h = synapses
    @unpack τ_syn_fast, τ_syn_slow = param
    
    @inbounds for i = 1:N
        Isyn[i] += dt * (-Isyn[i]/τ_syn_slow + h[i])
        h[i] += dt * (-h[i]/τ_syn_fast + spikes[i]/(τ_syn_fast*τ_syn_slow))
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

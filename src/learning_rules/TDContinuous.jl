#=
CriticNeuronのスパイクとrewardsからTD誤差を計算する
=#

# TDContinuousのパラメータ(固定)
@kwdef struct TDContinuousParameter{FT}
    τ_fast::FT = 50 # 早い時定数 [ms]
    τ_slow::FT = 200 # 遅い時定数 [ms]
    τ_reward::FT = 4000 # rewardsの減るスピード[ms]
    r0::FT = 2 # [reward units]*s scaling constant
    V0::FT = -40 # 
end

# TDContinuousの定義
@kwdef mutable struct TDContinuous{FT,UIT}
    param::TDContinuousParameter = TDContinuousParameter{FT}()
    N::UIT #シナプスの数
    # dx(t)/dt = -(1/τ_slow)*x(t) + h(t) + {(τ_slow-τ_fast)/(τ_slow*τ_fast)} * δi(t)  
    trace_vector::Vector{FT} = zeros(N) # 
    # dh(t)/dt = -(1/τ_fast)*h(t) + [(τ_fast-τ_slow)(τ_fast+τ_reward)/{τ_slow * (τ_fast)^2 * τ_reward}]*δi(t)
    h::Vector{FT} = zeros(N) # トレースの補助変数
    # δTD(t) = r0*sum(trace_vector)/N - V0/τ_reward + reward
    td_error::FT = 0
end

# TDContinuousに対するupdate!メソッドの定義
function update!(td::TDContinuous{FT,UIT}, param::TDContinuousParameter{FT}, dt::FT, spikes::BitVector, reward::FT) where {FT,UIT}
    @unpack N, trace_vector, h = td
    @unpack τ_fast, τ_slow, τ_reward, r0, V0 = param
    kτ_1::FT = (τ_slow - τ_fast)/(τ_slow*τ_fast)
    kτ_2::FT = kτ_1 * (τ_fast + τ_reward)/(τ_fast*τ_reward)
    @inbounds for i = 1:N
        trace_vector[i] += dt * (-trace_vector[i]/τ_slow + h[i] + spikes[i]*kτ_1/dt)
        h[i] += dt * (-h[i]/τ_fast - spikes[i]*kτ_2/dt)
    end
    td.td_error = (r0/N) * sum(trace_vector) - V0/τ_reward + reward
end

# TDContinuousに対するupdate!メソッドの定義
function update_threads!(td::TDContinuous{FT,UIT}, param::TDContinuousParameter{FT}, dt::FT, spikes::BitVector, reward::FT) where {FT,UIT}
    @unpack N, trace_vector, h = td
    @unpack τ_fast, τ_slow, τ_reward, r0, V0 = param
    kτ_1::FT = (τ_slow - τ_fast)/(τ_slow*τ_fast)
    kτ_2::FT = kτ_1 * (τ_fast + τ_reward)/(τ_fast*τ_reward)
    @inbounds Threads.@threads for i = 1:N
        trace_vector[i] += dt * (-trace_vector[i]/τ_slow + h[i] + spikes[i]*kτ_1/dt)
        h[i] += dt * (-h[i]/τ_fast - spikes[i]*kτ_2/dt)
    end
    td.td_error = (r0/N) * sum(trace_vector) - V0/τ_reward + reward
end


# TDContinuousに対するinit!メソッドの定義
function init!(td::TDContinuous)
    @unpack N, trace_vector, h, td_error = td
    trace_vector = zeros(N)
    h = zeros(N)
    td_error = 0
end

function get_trace_vector(td::TDContinuous{FT})::Vector{FT} where FT
    return td.trace_vector
end

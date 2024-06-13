#=
CriticNeuronのスパイクと入力されるシナプス電流からトレースを計算
∂V/∂wijに対応
=#

# LTPTraceのパラメータ(固定)
@kwdef struct LTPTraceParameter{FT}
    τ_fast::FT = 50 # 早い時定数 [ms]
    τ_slow::FT = 200 # 遅い時定数(膜の時定数と同じ？) [ms]
    τ_reward::FT = 4000 # rewardsの減るスピード[ms]
    Δu::FT = 2
    r0::FT = 2
end

# LTPTraceの定義
@kwdef mutable struct LTPTrace{FT,UIT}
    param::LTPTraceParameter = LTPTraceParameter{FT}()
    Npost::UIT #クリティックニューロンの数
    Npre::UIT #入力ニューロン数
    # dx(t)/dt = -(1/τ_slow)*x(t) + h(t) 
    trace_matrix::Matrix{FT} = zeros(Npost, Npre) # 
    # dh(t)/dt = -(1/τ_fast)*x(t) + [(τ_fast-τ_slow)(τ_fast+τ_reward)/{τ_slow * (τ_fast)^2 * τ_reward}]*δi(t)
    h::Matrix{FT} = zeros(Npost, Npre) # トレースの補助変数
    # ∂V_∂wij(t) = r0*sum(trace_matrix)/N - V0/τ_reward + reward
    ∂V_∂wij::Matrix{FT} = zeros(Npost, Npre)

    η::FT = 0.05 # critic:0.05, actor:0.0125
end

# LTPTraceに対するupdate!メソッドの定義
function update!(ltp_trace::LTPTrace{FT}, param::LTPTraceParameter{FT}, dt::FT, Isyn::Vector{FT}, critic_spikes::BitVector) where {FT}
    @unpack Npost, Npre, trace_matrix, h, η = ltp_trace
    @unpack τ_fast, τ_slow, τ_reward, r0, Δu = param

    @inbounds for j = 1:Npre
        @inbounds for i = 1:Npost
            trace_matrix[i,j] += dt * (-trace_matrix[i,j] / τ_slow + h[i,j])
            h[i,j] += dt * (-h[i,j]/τ_fast + Isyn[j]*critic_spikes[i]/(τ_slow*τ_fast))
        end
    end
    ltp_trace.∂V_∂wij = (η*r0 / (Npost*Δu)) * trace_matrix
end


# LTPTraceに対するinit!メソッドの定義
function init!(ltp_trace::LTPTrace)
    @unpack N, trace_matrix, h, ∂V_∂wij = ltp_trace
    trace_matrix::Matrix{FT} = zeros(Npost, Npre) # 
    h::Matrix{FT} = zeros(Npost, Npre)
    ∂V_∂wij::Matrix{FT} = zeros(Npost, Npre)
end

function get_trace_matrix(ltp_trace::LTPTrace{FT})::Matrix{FT} where {FT}
    return ltp_trace.trace_matrix
end

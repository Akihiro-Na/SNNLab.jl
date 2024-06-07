#=
    Spike2actionの定義ファイル
=#
# Spike2action のパラメータ(固定)
struct Spike2actionParameter{FT,UIT} <: AbstractEnvAgentInrerfaceParam{FT,UIT}
    actionset::Matrix{FT}
    function Spike2actionParameter{FT,UIT}(Naction::UIT) where {FT, UIT}
        actionset = zeros(2,Naction)
        for k in 1:Naction
            actionset[:,k] = [cos(2π*(k-1) / Naction),sin(2π*(k-1) / Naction)]
        end
        new{FT,UIT}(actionset)
    end
end

# Spike2action modelの定義
@kwdef mutable struct Spike2action{FT,UIT} <: AbstractEnvAgentInrerface{FT,UIT}
    Naction::UInt32
    param::Spike2actionParameter = Spike2actionParameter{FT,UIT}(Naction)
    action::Vector{FT} = [0,0]
end

# The case of 2D state 
function update!(s2a::Spike2action{FT,UIT}, param::Spike2actionParameter{FT,UIT}, Isyn::Vector{FT})::Vector{FT} where {FT,UIT}
    if sum(Isyn) ≠ 0
        s2a.action = (param.actionset * Isyn)/sum(Isyn)
    else
        s2a.action = [0,0]
    end
    return s2a.action
end

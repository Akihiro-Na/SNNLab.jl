#=
Spike2actionの定義ファイル
ニューロン集団のスパイク活動から２次元のアクションベクトルを生成

懸念点
・total_Isynで規格化していることにより神経活動が全体的に弱い場合でも同じ大きさのアクションが出てくる
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
    total_Isyn::FT = sum(Isyn)
    if total_Isyn ≠ 0
        s2a.action .= (param.actionset * Isyn) ./ total_Isyn
    else
        s2a.action .= FT(0)
    end
    return s2a.action
end

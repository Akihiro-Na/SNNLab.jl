
# state2λ のパラメータ(固定)
@kwdef struct State2λParameter{FT,UIT} <: AbstractEnvAgentInrerfaceParam{FT,UIT}
    λmax::FT = 400 # 最大の発火頻度 [Hz]
    σ::FT = 2 # place cell の配置間隔 [m]
    xmin::FT = -2
    xmax::FT = 22
    ymin::FT = -2
    ymax::FT = 22
    receptive_centers::Vector{Tuple{FT,FT}} = vec(collect(Iterators.product(xmin:σ:xmax, ymin:σ:ymax))) # 
    N::UIT = length(receptive_centers)
end

# state2λ modelの定義
@kwdef mutable struct State2λ{FT,UIT} <: AbstractEnvAgentInrerface{FT,UIT}
    param::State2λParameter = State2λParameter{FT,UIT}()
    λvec::Vector{FT} = zeros(FT, param.N)
    normvec::Vector{FT} = zeros(FT, param.N)  # norm保存用vector
end

function vector_norm_grid!(norm_grid::Vector{FT}, grid::Vector{Tuple{FT,FT}}, x, y)::Vector{FT} where FT
    #=
    for (i, (a, b)) in enumerate(grid)
        norm_grid[i] = norm([a - x, b - y])
    end
    =#
    @inbounds for i in eachindex(grid)
        a, b = grid[i]
        norm_grid[i] = sqrt((a - x)^2 + (b - y)^2)
    end
    return norm_grid
end

# The case of 2D state 
function update!(lambda::State2λ{FT,UIT}, param::State2λParameter{FT,UIT}, state::Tuple{FT, FT})::Vector{FT} where {FT,UIT}
    x, y = state
    @unpack λmax, σ, receptive_centers = param
    vector_norm_grid!(lambda.normvec, receptive_centers, x, y)
    for i in eachindex(lambda.normvec)
        lambda.λvec[i] = λmax * exp(-lambda.normvec[i]^2 / σ^2)
    end
    return lambda.λvec
end


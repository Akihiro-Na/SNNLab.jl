#=
    Spike2actionの定義ファイル
=#
# Spike2action のパラメータ(固定)
@kwdef struct Spike2actionParameter{FT} <: AbstractEnvAgentInrerfaceParam{FT}
    λmax::FT = 400 # 最大の発火頻度 [Hz]
    σ::FT = 2 # place cell の配置間隔 [m]
    xmin::FT = -2
    xmax::FT = 22
    ymin::FT = -2
    ymax::FT = 22
    receptive_centers::Vector{Tuple{FT,FT}} = vec(collect(Iterators.product(xmin:σ:xmax, ymin:σ:ymax))) # 
    N::UInt32 = length(receptive_centers)
end

# Spike2action modelの定義
@kwdef mutable struct Spike2action{FT} <: AbstractEnvAgentInrerface{FT}
    param::Spike2actionParameter = Spike2actionParameter{FT}()
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
function update!(lambda::Spike2action{FT}, param::Spike2actionParameter{FT}, state::Tuple{FT, FT})::Vector{FT} where FT
    x, y = state
    @unpack λmax, σ, receptive_centers = param
    vector_norm_grid!(lambda.normvec, receptive_centers, x, y)
    for i in eachindex(lambda.normvec)
        lambda.λvec[i] = λmax * exp(-lambda.normvec[i]^2 / σ^2)
    end
    return lambda.λvec
end

#=
old 
  0.272270 seconds (1.95 M allocations: 142.333 MiB, 53.03% gc time, 13.14% compilation time)
=#
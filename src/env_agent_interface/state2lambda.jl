
# state2λ のパラメータ(固定)
@kwdef struct State2λParameter{FT,UIT} <: AbstractEnvAgentInrerfaceParam{FT,UIT}
    λmax::FT = 400 # 最大の発火頻度 [Hz]
    σ::FT = 4 # place cell の配置間隔 [m]
    xinterval::FT = 2
    yinterval::FT = 2
    xmin::FT = -2
    xmax::FT = 22
    ymin::FT = -2
    ymax::FT = 22
    receptive_x::StepRangeLen{FT, Base.TwicePrecision{FT}, Base.TwicePrecision{FT}, UIT} = xmin:xinterval:xmax
    receptive_y::StepRangeLen{FT, Base.TwicePrecision{FT}, Base.TwicePrecision{FT}, UIT} = ymin:yinterval:ymax
    receptive_centers::Vector{Tuple{FT,FT}} = vec(collect(Iterators.product(receptive_x, receptive_y))) # 
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

# The case of 2D state for tuple state
function update!(lambda::State2λ{FT,UIT}, param::State2λParameter{FT,UIT}, state::Tuple{FT, FT})::Vector{FT} where {FT,UIT}
    x, y = state
    @unpack λmax, σ, receptive_centers = param
    vector_norm_grid!(lambda.normvec, receptive_centers, x, y)
    for i in eachindex(lambda.normvec)
        lambda.λvec[i] = λmax * exp(-lambda.normvec[i]^2 / σ^2)
    end
    return lambda.λvec
end

# The case of 2D state for vector state
function update!(lambda::State2λ{FT,UIT}, param::State2λParameter{FT,UIT}, state::Vector{FT})::Vector{FT} where {FT,UIT}
    @unpack λmax, σ, receptive_centers = param
    vector_norm_grid!(lambda.normvec, receptive_centers, state[1], state[2])
    for i in eachindex(lambda.normvec)
        lambda.λvec[i] = λmax * exp(-lambda.normvec[i]^2 / σ^2)
    end
    return lambda.λvec
end

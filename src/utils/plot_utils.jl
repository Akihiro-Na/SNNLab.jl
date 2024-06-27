#=
データを可視化する用の関数群
=#
using Interpolations # for CubicSplineInterpolation in Interp2D function 

# plot filled circle 
function plot_circle!(x, y, r; color=:red)
    θ = range(0, 2π, 100)
    xc = x .+ r .* cos.(θ)
    yc = y .+ r .* sin.(θ)
    plot!(xc, yc, seriestype=:shape, lw=2, color=color)
end

# plot not fild unit circle
function plot_unit_circle()
    # Plot unit circle
    θ = range(0, stop=2π, length=100)
    x = cos.(θ)
    y = sin.(θ)
    plot(x, y, aspect_ratio=:equal)
    #plot(x, y, label="Unit Circle", aspect_ratio=:equal, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
    #scatter!([0], [0], color=:black, label="Origin")
    scatter!([0], [0], color=:black)
end

##################################################################################
# ニューロンの発火トレースを2次元空間上の格子点の大きさで表現する関数(重ね書き) #########
##################################################################################
function plot_receptive_centers!(receptive_centers::Vector{Tuple{FT,FT}}, Isynarr::Matrix{FT}, timestep::UIT, title::String) where {FT,UIT}
    x_coords = [center[1] for center in receptive_centers]
    y_coords = [center[2] for center in receptive_centers]

    # 指定したタイムステップの Isyn 値を取得
    Isyn_values = Isynarr[timestep, :]

    # 点の大きさと色を設定
    sizes = abs.(Isyn_values) .* 1 .+ 0.1  # サイズのスケーリング（適宜調整）
    colors = Isyn_values  # 色のスケーリング

    scatter!(x_coords, y_coords, st=:scatter,
        marker_z=colors, markersize=sizes, c=:viridis, legend=false,
        xlabel="X", ylabel="Y",
        clims=(0, 10),
        title=title)
    # title = "timestep $timestep, td error = $td_error"
end

# ニューロンの発火トレースを2次元空間上の格子点の大きさで表現する関数(上書き) #########
function plot_receptive_centers(receptive_centers::Vector{Tuple{FT,FT}}, Isynarr::Matrix{FT}, timestep::UIT) where {FT,UIT}
    x_coords = [center[1] for center in receptive_centers]
    y_coords = [center[2] for center in receptive_centers]

    # 指定したタイムステップの Isyn 値を取得
    Isyn_values = Isynarr[timestep, :]

    # 点の大きさと色を設定
    sizes = abs.(Isyn_values) .* 1 .+ 0.2  # サイズのスケーリング（適宜調整）
    colors = Isyn_values  # 色のスケーリング

    p = scatter(x_coords, y_coords, st=:scatter,
        marker_z=colors, markersize=sizes, c=:viridis, legend=false,
        xlabel="X", ylabel="Y",
        clims=(0, 10),
        title="Receptive Centers at timestep $timestep")
        xmax = maximum(x_coords)
        xmin = minimum(x_coords)
    xlims!(p, (xmin-1, xmax+1))
end


##################################################################################
# ベクトル場のplot ################################################################
##################################################################################
# as: arrow head size 0-1 (fraction of arrow length;  la: arrow alpha transparency 0-1
# arrow0!は次のURLを参考https://discourse.julialang.org/t/plots-jl-arrows-style-in-quiver/13659/5
function arrow0!(x, y, u, v; as=0.1, lw=1, lc=:black, la=1)
    nuv = sqrt(u^2 + v^2)
    v1, v2 = [u; v] / nuv, [-v; u] / nuv
    v4 = (3 * v1 + v2) / 3.1623  # sqrt(10) to get unit vector
    v5 = v4 - 2 * (v4' * v2) * v2
    v4, v5 = as * nuv * v4, as * nuv * v5
    plot!([x, x + u], [y, y + v], lw=lw, lc=lc, la=la, label=false)
    plot!([x + u, x + u - v5[1]], [y + v, y + v - v5[2]], lw=lw, lc=lc, la=la, label=false)
    plot!([x + u, x + u - v4[1]], [y + v, y + v - v4[2]], lw=lw, lc=lc, la=la, label=false)
end

##################################################################################
# より滑らかなheatmapをplotするためのデータを補間で生成する関数 ######################
##################################################################################
function Interp2D(data, factor)
    IC = CubicSplineInterpolation((axes(data, 1), axes(data, 2)), data)

    finerx = LinRange(firstindex(data, 1), lastindex(data, 1), size(data, 1) * factor)
    finery = LinRange(firstindex(data, 2), lastindex(data, 2), size(data, 2) * factor)
    nx = length(finerx)
    ny = length(finery)

    data_interp = Array{Float64}(undef, nx, ny)
    for i ∈ 1:nx, j ∈ 1:ny
        data_interp[i, j] = IC(finerx[i], finery[j])
    end

    return finery, finerx, data_interp

end
#=
データを可視化する用の関数群
=#

function plot_circle!(x, y, r; color=:red)
    θ = range(0, 2π, 100)
    xc = x .+ r .* cos.(θ)
    yc = y .+ r .* sin.(θ)
    plot!(xc, yc, seriestype=:shape, lw=2, color=color)
end


# ニューロンの発火トレースを2次元空間上の格子点の大きさで表現する関数
function plot_receptive_centers!(receptive_centers::Vector{Tuple{FT,FT}}, Isynarr::Matrix{FT}, timestep::UIT) where {FT,UIT}
    x_coords = [center[1] for center in receptive_centers]
    y_coords = [center[2] for center in receptive_centers]

    # 指定したタイムステップの Isyn 値を取得
    Isyn_values = Isynarr[timestep, :]

    # 点の大きさと色を設定
    sizes = abs.(Isyn_values) .* 1  # サイズのスケーリング（適宜調整）
    colors = Isyn_values  # 色のスケーリング

    scatter!(x_coords, y_coords, size=(600, 600), st=:scatter,
        marker_z=colors, markersize=sizes, c=:viridis, legend=false,
        xlabel="X", ylabel="Y",
        clims=(0, 10),
        title="Receptive Centers at timestep $timestep")
end
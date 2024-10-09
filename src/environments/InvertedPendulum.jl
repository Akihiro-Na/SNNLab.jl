#=
Inverted Pendulum環境の定義ファイル
=#

# MazeParam構造体の定義(固定)
@kwdef struct InvertedPendulumParam{FT} <: AbstractEnvironmentParam{FT}
    length::FT = 1 # [m]
    mass::FT = 60 # [kg]
    state_ref::Vector{FT} = [0.0, 0.0] # [rad]
    gravity::FT = 9.81 # [m/s^2] 重力加速度
    max_torque::FT = 10.0 # [Nm] 最大トルク
    moment_of_inertia::FT = mass*length^2 # 慣性モーメント
    mgh::FT = mass*gravity*length # 
    # 受動的粘弾性
    K::FT = mgh*0.8
    B::FT = 4
    sigma::FT = 0.2
    dt::FT = dt #ms
    dt_sec::FT = dt*10^(-3) #ms -> sec
end

# Maze構造体の定義
@kwdef mutable struct InvertedPendulum{FT} <: AbstractEnvironment{FT}
    param::InvertedPendulumParam{FT} = InvertedPendulumParam{FT}()
    state::Vector{FT} = [0.0, 0.0] # [θ, dθ/dt] 振り子の角度と角速度
    reward::FT = 0.0
end


# 環境の状態更新関数（actionはトルク[Nm]）
function update!(pendulum::InvertedPendulum{FT}, action::FT) where FT
    θ, dθdt = pendulum.state
    param = pendulum.param
    
    # 制約されたトルク
    torque = clamp(action, -param.max_torque, param.max_torque)

    # 運動方程式（倒立振子の角加速度）
    ddθdt = (param.mgh * sin(θ) - param.K * θ - param.B * dθdt + torque) / (param.moment_of_inertia)

    # 状態の更新（オイラー法で数値積分）
    θ += dθdt * param.dt_sec
    dθdt += ddθdt * param.dt_sec + param.sigma * randn() * sqrt(param.dt_sec)

    # 状態を保存
    pendulum.state[1] = θ
    pendulum.state[2] = dθdt

    # 報酬の計算（目標状態に基づく）
    θ_ref, dθdt_ref = param.state_ref
    error_θ = θ - θ_ref
    error_dθdt = dθdt - dθdt_ref
    pendulum.reward = - (abs(error_θ) + abs(error_dθdt)) # 角度と角速度の誤差を反映した報酬

end

# 倒立振子の初期化関数
function init!(pendulum::InvertedPendulum{FT}, init_state::Vector{FT}) where FT
    pendulum.state = init_state
end

# 倒立振子の状態を取得する関数
function get_state(pendulum::InvertedPendulum{FT})::Vector{FT} where FT
    return pendulum.state
end

# 変数の保存
# 状態と時刻を記録するための構造体
mutable struct InvertedPendulumLog{FT}
    time::Vector{FT}
    state_history::Matrix{FT}

    # コンストラクタで事前に確保する要素数を指定
    function InvertedPendulumLog{FT}(capacity::UIT) where {FT, UIT}
        # コンストラクタ内でメモリを確保
        time = Vector{FT}(undef, capacity)
        state_history = zeros(FT, capacity, 2)  # 各要素を初期化
        new{FT}(time, state_history)
    end
end

# 状態と時刻を保存する関数（push! を使わず、インデックス指定で代入）
function save_log!(log::InvertedPendulumLog{FT}, pendulum::InvertedPendulum{FT}, time::FT, index::UIT) where {FT, UIT}
    log.time[index] = time
    log.state_history[index,:] = pendulum.state
end

####################################################
# 以下プロット用の関数群 ###############################
####################################################

# 振子の姿勢をプロットする関数
function plot_pendulum(θ::FT, length::FT) where FT
    # 振子の先端の位置を計算
    x = length * sin(θ)  # x座標
    y = -length * cos(θ)  # y座標 (重力の方向に向ける)

    # 振子の棒をプロット
    plot([0, x], [0, y], lw=2, label="", xlabel="x", ylabel="y", xlims=(-length, length), ylims=(-length, length), size=(400, 400))
    scatter!([x], [y], color=:red, label="Mass", markersize=8)  # 振子の端に重りを描画
end

# 状態の時系列をプロットする関数
function plot_timeseries(log::InvertedPendulumLog{FT}) where FT
    plot(log.time/(10^3), log.state_history[:, 1], label="θ (Angle)", xlabel="Time [s]", ylabel="State", lw=2)
    plot!(log.time/(10^3), log.state_history[:, 2], label="dθ/dt (Angular Velocity)", lw=2)
end


# 振子の動きをアニメーション表示する関数
function animate_pendulum(log::InvertedPendulumLog{FT}, length::FT, filename::String) where FT
    anim = @animate for i in 1:length(log.time)
        plot_pendulum(log.state_history[i, 1], length)
    end
    gif(anim, filename, fps=30)
end
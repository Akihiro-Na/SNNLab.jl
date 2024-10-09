#=
InvertedPendulum環境のtest用コード
=#
using SNNLab
using Plots

function run_inverted_pendulum_test()

    FT = Float32
    UIT = UInt32

    T = 60* 10^3 # ms
    dt::FT = 1 # ms
    nt = UIT(T / dt) # number of timesteps
    t = Array{FT}(1:nt) * dt


    # modelの定義
    param_env = SNNLab.InvertedPendulumParam{FT}(
        length=1.0f0, 
        mass=1.0f0, 
        state_ref=[0.0f0, 0.0f0],
        dt = dt
    )
    env = InvertedPendulum{FT}(param=param_env, state=[0.3, 0.0]) # 初期状態: [θ, dθ/dt] = [0.1, 0.0]
  

    # 記録用
    log_state = SNNLab.InvertedPendulumLog{FT}(nt)

    # シミュレーション
    @time for i = 1:nt
        action::FT = 2 * rand() - 1 # ランダムトルク [-1, 1]
        update!(env, action)
        save_log!(log_state, env, i * env.param.dt, i) 
    end

    # アニメーションのインスタンス生成
    anim = Animation()
    ylim = (-1.5, 1.5) # プロットのy軸の範囲
    # アニメーションのプロット
    for i = 1:nt
        if mod(i, 100) == 0 # アニメーションのフレームを減らすために50ステップごとにプロット
            θ = log_state.state_history[i,1]
            plot([0, sin(θ)], [0, cos(θ)], size=(400, 400), st=:line, lw=3, label="Pendulum")
            scatter!([sin(θ)], [cos(θ)], color=:red, label="Mass")
            ylims!(ylim)
            xlims!(ylim)
            frame(anim)
        end
    end

    # GIFとして保存
    gif(anim, "inverted_pendulum.gif", fps=10)

    SNNLab.plot_timeseries(log_state)
    
end

run_inverted_pendulum_test()
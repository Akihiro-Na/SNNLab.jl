#=
Maze環境のtest用コード
=#
using SNNLab
using Plots

function run_Maze_test()

    function plot_circle!(x, y, r; color=:red)
        θ = range(0, 2π, 100)
        xc = x .+ r .* cos.(θ)
        yc = y .+ r .* sin.(θ)
        plot!(xc, yc, seriestype=:shape, lw=2, color=color)
    end

    FT = Float32

    T = 10 * 10^3 # ms
    dt::FT = 1 # ms
    nt = UInt32(T / dt) # number of timesteps
    t = Array{FT}(1:nt) * dt


    # modelの定義
    env = Maze{FT}(start=[2, 2])
    #init!(env, (1,1),0)

    # 記録用

    statearr = zeros(FT, nt, 2)

    # simulation
    #アニメーションのインスタンス生成
    anim = Animation()
    xylim = (0, 20)
    param = env.param
    @time for i = 1:nt
        action::FT = 2 * rand() - 1 # random action
        update!(env, param, action, dt)
        statearr[i, :] = env.state
    end

    #plot animatin
    for i = 1:nt
        if mod(i, 100) == 0
            plot([statearr[i, 1]], [statearr[i, 2]], size=(250, 250), st=:scatter,
                xlims=xylim, ylims=xylim)
            plt = plot_circle!(param.goal[1], param.goal[2], param.goal_radius; color=:red)
            frame(anim, plt)
        end
    end
    gif(anim, "maze.gif", fps=10)

end
#@time 
#@code_warntype
run_Maze_test()
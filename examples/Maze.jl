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

    FT = Float64

    T = 100 # ms
    dt::FT = 0.01 # ms
    nt = UInt32(T / dt) # number of timesteps
    t = Array{FT}(1:nt) * dt


    # modelの定義
    env = Maze{FT}(start=[1, 1])
    #init!(env, (1,1),0)


    # simulation
    #アニメーションのインスタンス生成
    anim = Animation()
    xylim = (0, 20)
    param = env.param
    @time for i = 1:nt
        action = 2 * rand() - 1 # random action
        update!(env, param, action, dt)
        #plot animatin
        plot([env.state[1]], [env.state[2]], size=(250, 250), st=:scatter,
            xlims=xylim, ylims=xylim)
        plt = plot_circle!(param.goal[1], param.goal[2], param.goal_radius; color=:red)
        if mod(i, 1000) == 0
            frame(anim, plt)
        end
    end
    gif(anim, "maze.gif", fps=10)

end
run_Maze_test()
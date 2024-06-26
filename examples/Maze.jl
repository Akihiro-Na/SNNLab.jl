#=
Maze環境のtest用コード
=#
using SNNLab
using Plots

function run_Maze_test()

    FT = Float32

    T = 60 * 10^3 # ms
    dt::FT = 1 # ms
    nt = UInt32(T / dt) # number of timesteps
    t = Array{FT}(1:nt) * dt


    # modelの定義
    param_env = SNNLab.MazeParam{FT}(velocity=50*10^-3)
    env = Maze{FT}(start=[5, 5], param=param_env)
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
            plt = SNNLab.plot_circle!(param.goal[1], param.goal[2], param.goal_radius; color=:red)
            frame(anim, plt)
        end
    end
    gif(anim, "maze.gif", fps=10)

end
#@time 
#@code_warntype
run_Maze_test()
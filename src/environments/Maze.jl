#=
Maze環境の定義ファイル
=#

# MazeParam構造体の定義(固定)
@kwdef struct MazeParam{FT} <: AbstractEnvironmentParam{FT}
    hight::FT = 20 # [m]
    width::FT = 20 # [m]
    goal::Vector{FT} = [10,10] # ゴール地点
    goal_radius::FT = 1 # [m]
    velocity::FT = 10*10^-3 # [m/ms]
    # obstacles::Vector{Tuple{FT, FT, FT, FT}}
end

# Maze構造体の定義
@kwdef mutable struct Maze{FT} <: AbstractEnvironment{FT}
    param::MazeParam = MazeParam{FT}()
    start::Vector{FT}  # スタート地点
    state::Vector{FT} = start # エージェントの位置
    next_state::Vector{FT} = start
    reward::FT = 0
end

# 環境のupdate!関数 actionは[-1,1]
function update!(maze::Maze{FT}, param::MazeParam, action::FT, dt::FT) where FT
    dx = param.velocity * cospi(action) * dt
    dy = param.velocity * sinpi(action) * dt
    maze.next_state = [maze.state[1] + dx, maze.state[2] + dy]
    if is_valid_move(maze.next_state, param)
        maze.state = maze.next_state
    else
        return maze.reward = -1.0
    end

    if is_goal_reached(maze,param)
        return maze.reward = 100.0
    else
        return maze.reward = 0.0
    end
end

# 環境のupdate!関数 actionは[-1,1]
function update!(maze::Maze{FT}, param::MazeParam, action::Vector{FT}, dt::FT) where FT
    dx = param.velocity * action[1] * dt
    dy = param.velocity * action[2] * dt
    maze.next_state = [maze.state[1] + dx, maze.state[2] + dy]
    if is_valid_move(maze.next_state, param)
        maze.state = maze.next_state
    else
        return maze.reward = -1.0
    end

    if is_goal_reached(maze,param)
        return maze.reward = 100.0
    else
        return maze.reward = 0.0
    end
end

# 迷路の初期化関数 ---------------------------------------
function init!(maze::Maze, start::Vector{FT}, num_obstacles::Int) where FT
    # 初期化パラメータ
    maze.start = start
    maze.state = start
    maze.next_state = start
    
    # ランダムな障害物を生成
    #=
    for _ in 1:num_obstacles
        obstacle_width = rand() * maze.param.width * 0.1
        obstacle_height = rand() * maze.param.height * 0.1
        obstacle_x = rand() * (maze.param.width - obstacle_width)
        obstacle_y = rand() * (maze.param.height - obstacle_height)
        push!(maze.param.obstacles, (obstacle_x, obstacle_y, obstacle_width, obstacle_height))
    end
    =#
end

# 有効な移動かどうかをチェックする関数 ----------------------
function is_valid_move(mazestate::Vector{FT}, param::MazeParam)::Bool where FT
    x = mazestate[1]
    y = mazestate[2]
    # 迷路の境界チェック
    if x < 0 || x > param.width || y < 0 || y > param.hight
        return false
    end
    # 障害物との衝突チェック
    #=
    for obs in maze.obstacles
        if (x > obs[1] && x < obs[1] + obs[3]) && (y > obs[2] && y < obs[2] + obs[4])
            return false
        end
    end
    =#
    return true
end

# 現在のエージェントの状態を取得する関数 ----------------------
function get_state(maze::Maze)::Vector{FT}
    return maze.state
end

# ゴールに到達したかどうかをチェックする関数 -------------------
function is_goal_reached(maze::Maze, param::MazeParam)::Bool
    x = maze.state[1]
    y = maze.state[2]
    x_goal, y_goal = param.goal
    goal_radius = param.goal_radius
    if (x - x_goal)^2 + (y - y_goal)^2 < goal_radius^2
        return true
    end

    return false
end


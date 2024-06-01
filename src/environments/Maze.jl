#=
Maze環境の定義ファイル
=#

# MazeParam構造体の定義(固定)
struct MazeParam <: AbstractEnvironmentParam{FT}
    hight::FT = 20 # [m]
    width::FT = 20 # [m]
    goal::Tuple{FT, FT} = (10,10) # ゴール地点
    goal_radius::FT = 1 # [m]
    velocity = 1 # [m/s]
    obstacles::Vector{Tuple{FT, FT, FT, FT}}
end

# Maze構造体の定義
@kwdef mutable struct Maze <: AbstractEnvironment{FT}
    param::MazeParam = MazeParam{FT}()
    start::Tuple{FT, FT}  # スタート地点
    state::Tuple{FT, FT}  # エージェントの位置
end

# 環境のupdate!関数 actionは[-1,1]
function update!(maze::Maze, param::MazeParam, action::Float64, dt)
    x, y = maze.state
    dx = param.velocity * cospi(action) * dt
    dy = param.velocity * sinpi(action) * dt
    state_position = (x + dx, y + dy)
    if is_valid_move(state_position, maze)
        maze.state = state_position
    else
        reward = -2.0
    end

    if is_goal_reached(state_position)
        reward = 2.0
    else
        reward = 0.0
    end

    return reward
end

# 迷路の初期化関数 ---------------------------------------
function init!(maze::Maze, start::Tuple{FT, FT}, num_obstacles::Int) where FT
    # 初期化パラメータ
    maze.start = start
    maze.state = start
    
    # ランダムな障害物を生成
    for _ in 1:num_obstacles
        obstacle_width = rand() * maze.param.width * 0.1
        obstacle_height = rand() * maze.param.height * 0.1
        obstacle_x = rand() * (maze.param.width - obstacle_width)
        obstacle_y = rand() * (maze.param.height - obstacle_height)
        push!(maze.param.obstacles, (obstacle_x, obstacle_y, obstacle_width, obstacle_height))
    end
end

# 有効な移動かどうかをチェックする関数 ----------------------
function is_valid_move(maze::Maze, param::MazeParam)::Bool
    x, y = maze.state
    # 迷路の境界チェック
    if x < 0 || x > param.width || y < 0 || y > param.height
        return false
    end
    # 障害物との衝突チェック
    for obs in maze.obstacles
        if (x > obs[1] && x < obs[1] + obs[3]) && (y > obs[2] && y < obs[2] + obs[4])
            return false
        end
    end
    return true
end

# 現在のエージェントの状態を取得する関数 ----------------------
function get_state(maze::Maze)::Tuple{Int, Int}
    return maze.state
end

# ゴールに到達したかどうかをチェックする関数 -------------------
function is_goal_reached(maze::Maze, param::MazeParam)::Bool
    x,y = maze.state
    x_goal, y_goal = param.goal
    goal_radius = param.goal_radius
    if (x - x_goal)^2 + (y - y_goal)^2 < goal_radius^2
        return true
    end

    return false
end


using SNNLab
using Test

############################################
## Tests for Neuron Model ##################
############################################
@testset "LIFNeuron Tests" begin
    # LIFNeuronファイル内の関数や構造体のテスト
    include("../examples/LIFNeuron.jl")
end

@testset "PoissonNeuron Tests" begin
    # PoissonNeuronファイル内の関数や構造体のテスト
    include("../examples/PoissonNeuron.jl")
end

############################################
## Tests for Synapse Model #################
############################################
@testset "DoubleExpSynapse Tests with LIF" begin
    include("../examples/DoubleExpSynapse.jl")
end

############################################
## Tests for Environment Model #############
############################################
@testset "Maze Env Tests" begin
    # Mazeファイル内の関数や構造体のテスト
    include("../examples/Maze.jl")
end

############################################
## Tests for env_agent_interface  ##########
############################################
@testset "state2λ Tests" begin
    include("../examples/state2lambda.jl")
end

@testset "spike2action Tests" begin
    include("../examples/spike2action.jl")
end

############################################
## Tests Learning through interaction ######
############################################
@testset "Continuous Maze learning with TD LTP Tests" begin
    include("../examples/Nicolas_etal_2013.jl")
end
using SNNLab
using Test

@testset "SNNLab.jl" begin
    # Write your tests here.
    
end

@testset "LIFNeuron Tests" begin
    module LIFNeuronTests
        # LIFNeuronファイル内の関数や構造体のテスト
        include("../examples/LIFNeuron.jl")
    end
end

@testset "PoissonNeuron Tests" begin
    module PoissonNeuronTests
        # PoissonNeuronファイル内の関数や構造体のテスト
        include("../examples/PoissonNeuron.jl")
    end
end

@testset "Maze Env Tests" begin
    module MazeEnvTests
        # Mazeファイル内の関数や構造体のテスト
        include("../examples/Maze.jl")
    end
end

@testset "DoubleExpSynapse Tests with LIF" begin
    module DoubleExpSynapseTests
        include("../examples/DoubleExpSynapse.jl")
    end
end

@testset "state2λ" begin
    module State2LambdaTests
        include("../examples/state2lambda.jl")
    end
end
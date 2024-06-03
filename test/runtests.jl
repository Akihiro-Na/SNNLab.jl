using SNNLab
using Test

@testset "SNNLab.jl" begin
    # Write your tests here.
    
end

@testset "LIFNeuron Tests" begin
    # LIFNeuronモジュール内の関数や構造体のテスト
    include("../examples/LIFNeuron.jl")
end

@testset "PoissonNeuron Tests" begin
    # LIFNeuronモジュール内の関数や構造体のテスト
    include("../examples/PoissonNeuron.jl")
end

@testset "Maze Env Tests" begin
    # LIFNeuronモジュール内の関数や構造体のテスト
    include("../examples/Maze.jl")
end

@testset "DoubleExpSynapse Tests with LIF" begin
    include("../examples/DoubleExpSynapse.jl")
end
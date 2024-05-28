using SNNLab
using Test

@testset "SNNLab.jl" begin
    # Write your tests here.
    
end

@testset "LIFNeuron Tests" begin
    # LIFNeuronモジュール内の関数や構造体のテスト

    # 例えばLIFNeuronモジュールに LIFNeuron 構造体があり、コンストラクタに引数を渡す場合
    neurons = LIF{Float32}(N=2)
    @test neurons.N == 2

    @testset "LIFNeuron's method" begin
        # update! メソッドが実装されているかのテスト
        @test isdefined(neurons, :update!)
        # get_spike メソッドが実装されているかのテスト
        @test isdefined(neurons, :get_spike)
        # get_v メソッドが実装されているかのテスト
        @test isdefined(neurons, :get_v)
    end
end
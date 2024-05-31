using SNNLab
using Test

@testset "SNNLab.jl" begin
    # Write your tests here.
    
end

@testset "LIFNeuron Tests" begin
    # LIFNeuronモジュール内の関数や構造体のテスト

    # 例えばLIFNeuronモジュールに LIFNeuron 構造体があり、コンストラクタに引数を渡す場合
    N = 2
    neurons = LIF{Float32}(N=N)
    @test neurons.N == N

    @testset "LIFNeuron's method" begin
        # update! メソッドが実装されているかのテスト
        @test isdefined(SNNLab, :update!)
        # get_spike メソッドが実装されているかのテスト
        @test isdefined(SNNLab, :get_spike)
        # get_v メソッドが実装されているかのテスト
        @test isdefined(SNNLab, :get_v)
    end
end
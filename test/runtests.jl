using SNNLab
using Test

@testset "SNNLab.jl" begin
    # Write your tests here.
    
end

@testset "LIFNeuron Tests" begin
    # LIFNeuronモジュール内の関数や構造体のテスト
    # update! メソッドが実装されているかのテスト
    @test isdefined(LIF, :update!)
    # get_spike メソッドが実装されているかのテスト
    @test isdefined(LIF, :get_spike)
    # get_v メソッドが実装されているかのテスト
    @test isdefined(LIF, :get_v)

    # 例えばLIFNeuronモジュールに LIFNeuron 構造体があり、コンストラクタに引数を渡す場合
    N=2
    neurons = LIF{Float32}(N=N)
    @test neurons.N == N
end
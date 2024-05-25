# src/neurons/neuron.jl
module Neuron

# 他のファイルでこのモジュールを使用する際に `AbstractNeuron` 型をエクスポート
export AbstractNeuron

# 抽象型 `AbstractNeuron` の定義。この型は具象クラス（具体的なニューロンモデル）の基底クラスとして機能する
abstract type AbstractNeuron end

"""
    AbstractNeuron

Abstract base class for neurons. Defines common properties and methods for all neuron models.
"""
# 可変構造体 `AbstractNeuron` の定義。ニューロンの共通プロパティを持つ基底クラス
mutable struct AbstractNeuron
    # 膜電位を表すプロパティ
    membrane_potential::Float64
    # 入力電流を表すプロパティ
    input_current::Float64
    # その他のパラメータを格納する辞書
    parameters::Dict{Symbol, Float64}
    
    # コンストラクタの定義
    function AbstractNeuron(; membrane_potential=0.0, input_current=0.0, parameters=Dict{Symbol, Float64}())
        # 新しい `AbstractNeuron` インスタンスを初期化
        return new(membrane_potential, input_current, parameters)
    end
end

"""
    step!(neuron::AbstractNeuron, dt::Float64)

Abstract method to update the neuron's state. To be implemented by specific neuron models.
"""
# ニューロンの状態を更新するための抽象メソッド
function step!(neuron::AbstractNeuron, dt::Float64)
    # メソッドエラーを投げることで、このメソッドが具象クラスで実装されるべきことを示す
    throw(MethodError(step!, (neuron, dt)))
end

"""
    reset!(neuron::AbstractNeuron)

Resets the neuron's state to its initial conditions.
"""
# ニューロンの状態を初期条件にリセットするメソッド
function reset!(neuron::AbstractNeuron)
    # 膜電位を初期状態（0.0）にリセット
    neuron.membrane_potential = 0.0
    # 入力電流を初期状態（0.0）にリセット
    neuron.input_current = 0.0
end

end # module Neuron
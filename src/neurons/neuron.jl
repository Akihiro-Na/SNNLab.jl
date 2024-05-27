# src/neurons/neuron.jl
module Neuron

# 抽象型の定義
abstract type AbstractNeuron{FT} end

abstract type AbstractNeuronParam{FT} end

# update! メソッドの抽象定義
function update!(neuron::AbstractNeuron{FT}, param::AbstractNeuronParam{FT}, dt::FT, Ie::Vector{FT}) where FT
	throw(MethodError(update!, (neuron, param, dt, Ie)))
end

function get_spike(neuron::AbstractNeuron)::Vector{Bool}
    throw(MethodError(get_spike,()))
end

function get_v(neuron::AbstractNeuron{FT})::Vector{FT} where FT
    throw(MethodError(get_v,()))
end

end # module Neuron
# src/synapses/AbSynapse.jl

# 抽象型の定義
abstract type AbstractSynapse{FT} end

abstract type AbstractSynapseParam{FT} end

# update! メソッドの抽象定義
function update!(synapse::AbstractSynapse{FT}, param::AbstractSynapseParam{FT}, dt::FT, spikes::Vector{Bool}) where FT
	throw(MethodError(update!, (synapse, param, dt, spikes)))
end

# init! メソッドの抽象定義
function init!(synapse::AbstractSynapse)
	throw(MethodError(init!, (synapse)))
end

function get_Isyn(synapse::AbstractSynapse{FT})::Vector{FT} where FT
    throw(MethodError(get_Isyn,()))
end

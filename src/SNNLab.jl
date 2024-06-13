module SNNLab

# using
using LinearAlgebra
using Plots
using Random
using ProgressBars # for progress bar

# using at define of neurons -----
using Base: @kwdef
using Parameters: @unpack # or using UnPack
# --------------------------------

# neuron models =================
include("neurons/AbNeuron.jl")
include("neurons/LIFNeuron.jl")
include("neurons/PoissonNeuron.jl")
# end of neuron model ==========

# synapse models =================
include("synapses/AbSynapse.jl")
include("synapses/DoubleExpSynapse.jl")
# end of synapse model ==========

# leraning_rule =======================
include("learning_rules/LTPTrace.jl")
include("learning_rules/TDContinuous.jl")
# =====================================

# network =============================
include("networks/3LayerActorCritic.jl")
# =====================================

# env_agent_interface =================
include("env_agent_interface/AbEnvAgentInterface.jl")
include("env_agent_interface/state2lambda.jl")
include("env_agent_interface/Spike2action.jl")
# end of env_agent_interface ==========

# agent models =================
include("agents/AgentTDLTP.jl")
# end of synapse model ==========

# environment models =================
include("environments/AbEnvironment.jl")
include("environments/Maze.jl")
# end of synapse model ==========





# Neuron model
export LIF, PPPNeuron
# Synapse model
export DExpSynapse
# leraning_rule
export TDContinuous, LTPTrace
# networks
export L3ActorCritic
# env_agent_interface
export State2λ,Spike2action
# Agent model
export TDLTPAgent
# Environment model
export Maze

# Multiple dispatch function
export update!,init!


end # module

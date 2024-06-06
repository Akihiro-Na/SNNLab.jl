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

# environment models =================
include("environments/AbEnvironment.jl")
include("environments/Maze.jl")
# end of synapse model ==========

# env_agent_interface =================
include("env_agent_interface/AbEnvAgentInterface.jl")
include("env_agent_interface/state2lambda.jl")
# end of env_agent_interface ==========

# Neuron model
export LIF, PPPNeuron
# Synapse model
export DExpSynapse
# Environment model
export Maze
# env_agent_interface
export State2λ

# Multiple dispatch function
export update!,init!


end # module

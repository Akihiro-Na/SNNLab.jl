module SNNLab

# using
using LinearAlgebra
using Plots
using Random

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

# Neuron model
export LIF, PPPNeuron
# Synapse model
export DExpSynapse
# Environment model
export Maze
# Multiple dispatch function
export update!,init!

end # module

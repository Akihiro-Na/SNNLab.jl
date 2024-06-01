module SNNLab

# using
using LinearAlgebra
using Plots
using Random

# using at define of neurons -----
using Base: @kwdef
using Parameters: @unpack # or using UnPack
# --------------------------------

# neuron model =================
include("neurons/AbNeuron.jl")
include("neurons/LIFNeuron.jl")
include("neurons/PoissonNeuron.jl")
# end of neuron model ==========

# synapse model =================
include("synapses/AbSynapse.jl")
include("synapses/DoubleExpSynapse.jl")
# end of synapse model ==========

export LIF, PPPNeuron
export DExpSynapse
export Maze
export update!,init!

end # module

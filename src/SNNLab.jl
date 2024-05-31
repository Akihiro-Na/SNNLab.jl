module SNNLab

# using
using LinearAlgebra

include("neurons/AbNeuron.jl")
include("neurons/LIFNeuron.jl")

include("synapses/AbSynapse.jl")
include("synapses/DoubleExpSynapse.jl")

export LIF
export DExpSynapse

end # module

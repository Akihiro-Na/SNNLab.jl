#=
TDLTP用エージェントの定義ファイル
@time update!(agents.lambda, agents.lambda.param, state)
    @time update!(agents.network, agents.lambda.λvec, dt, reward)
    @time update!(agents.s2a,agents.s2a.param,agents.network.actor_synapses.Isyn)
 0.000009 seconds
   0.000328 seconds (25 allocations: 544 bytes)
  0.000010 seconds (2 allocations: 144 bytes)  
=#

# TDLTPAgentニューロンのパラメータ(固定)
@kwdef struct TDLTPAgentParameter{FT}
    
end

# TDLTPAgentニューロンの定義
@kwdef mutable struct TDLTPAgent{FT,UIT}
    Ncritic::UIT
    Nactor::UIT
    nt::UIT

    # state2lambdaの定義 ===
    lambda = State2λ{FT,UIT}()
    Ninput::UIT = lambda.param.N
    # ===============================

    network = L3ActorCritic{FT,UIT}(Ninput=Ninput,Ncritic=Ncritic,Nactor=Nactor,nt=nt)
    
    # spike2actionの定義 ===
    s2a = Spike2action{FT,UIT}(Naction=Nactor)
    # ===============================
    
end


# TDLTPAgentNeuronに対するupdate!メソッドの定義
function update!(agents::TDLTPAgent{FT,UIT}, dt::FT, state::Vector{FT}, reward::FT) where {FT,UIT}
    update!(agents.lambda, agents.lambda.param, state)
    update!(agents.network, agents.lambda.λvec, dt, reward)
    #update_threads!(agents.network, agents.lambda.λvec, dt, reward)
    update!(agents.s2a,agents.s2a.param,agents.network.actor_synapses.Isyn)
end

# TDLTPAgentNeuronに対するinit!メソッドの定義
function init!(agents::TDLTPAgent)
    
end
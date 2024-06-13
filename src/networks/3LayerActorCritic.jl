#=
L3ActorCritic networkの定義ファイル
=#

# L3ActorCritic networkのパラメータ(固定)
@kwdef struct L3ActorCriticParameter{FT}
    
end



# L3ActorCritic networkの定義
@kwdef mutable struct L3ActorCritic{FT,UIT}
    Ninput::UIT
    Ncritic::UIT
    Nactor::UIT
    nt::UIT

    # PoissonNeuronの定義 ===========
    input_neurons = PPPNeuron{FT}(N=Ninput, nt=nt)
    input_synapses = DExpSynapse{FT}(N=Ninput)
    # ===============================

    # CriticNeuronの定義 =============
    critic_neurons = LIF{FT}(N=Ncritic)
    critic_synapses = DExpSynapse{FT}(N=Ncritic)
    # 隣接行列
    w_input2critic::Matrix{FT} = rand(Ncritic,Ninput)*5
    Ie_i2c::Vector{FT} = w_input2critic * input_synapses.Isyn
    # TD誤差と学習用トレース
    td = TDContinuous{FT,UIT}(N=Ncritic)
    critic_ltp = LTPTrace{FT,UIT}(Npost=Ncritic ,Npre=Ninput)
    # ===============================

    # ActorNeuronの定義 ===============
    actor_neurons = LIF{FT}(N=Nactor)
    actor_synapses = DExpSynapse{FT}(N=Nactor)
    # 隣接行列 input to actor
    w_input2actor::Matrix{FT} = rand(Nactor,Ninput)
    # actor to actor
    w_actor2actor::Matrix{FT} = zeros(Nactor,Nactor)
    
    Ie_i2a::Vector{FT} = w_input2actor * input_synapses.Isyn
    # 学習用トレース
    actor_ltp = LTPTrace{FT,UIT}(Npost=Nactor ,Npre=Ninput)
    # ===============================
end

# L3ActorCriticNeuronに対するupdate!メソッドの定義
function update!(network::L3ActorCritic{FT,UIT}, λvec::Vector{FT}, dt::FT, reward::FT) where {FT, UIT}
    # input neuron ================================
    update!(network.input_neurons, dt, λvec)
    # synapse
    update!(network.input_synapses, network.input_synapses.param, dt, network.input_neurons.spike)
    # ==============================================

    # critic_neurons ================================
    network.Ie_i2c = network.w_input2critic * network.input_synapses.Isyn
    update!(network.critic_neurons, network.critic_neurons.param, dt, network.Ie_i2c)
    # synapse
    update!(network.critic_synapses, network.critic_synapses.param, dt, network.critic_neurons.spike)
    # TD-error
    update!(network.td, network.td.param, dt, network.critic_neurons.spike, reward)
    # LTPTrace
    update!(network.critic_ltp, network.critic_ltp.param, dt, network.input_synapses.Isyn, network.critic_neurons.spike)
    # w update
    network.w_input2critic += network.td.td_error * network.critic_ltp.∂V_∂wij *dt
    # ===============================================

    # actor_neurons ================================
    network.Ie_i2a = network.w_input2actor * network.input_synapses.Isyn + network.w_actor2actor * network.actor_synapses.Isyn
    update!(network.actor_neurons, network.actor_neurons.param, dt, network.Ie_i2a)
    # synapse
    update!(network.actor_synapses, network.actor_synapses.param, dt, network.actor_neurons.spike)
    # LTPTrace
    update!(network.actor_ltp, network.actor_ltp.param, dt, network.input_synapses.Isyn, network.actor_neurons.spike)
    # w update
    network.w_input2actor += network.td.td_error * network.actor_ltp.∂V_∂wij *dt
    # ===============================================

    return network.actor_synapses.Isyn
end


# L3ActorCriticNeuronに対するinit!メソッドの定義
function init!(network::L3ActorCritic)
    network.w_input2critic = rand(network.Ncritic,network.Ninput)*5
    network.w_input2actor = rand(network.Nactor,network.Ninput)
    function fa2a(x,Nactor)
        f = exp(-((x-ceil(Nactor/2))/2)^2)
    end
    wm = -60
    wp = 30
    fvec = fa2a.(1:network.Nactor,network.Nactor)
    wvec = wm/network.Nactor .+ wp*fvec/sum(fvec)
    for i in 1:network.Nactor
        network.w_actor2actor[i,:] = circshift(wvec, i-ceil(network.Nactor/2))
    end
end
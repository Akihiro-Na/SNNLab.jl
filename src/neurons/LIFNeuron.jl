#=
LIFニューロンの定義ファイル
=#

# LIFニューロンのパラメータ(固定)
@kwdef struct LIFParameter{FT} <: AbstractNeuronParam{FT}
    tref::FT = 2 # 不応期 [ms]
    tau_m::FT = 10 # 膜時定数 [ms]
    vrest::FT = -60 # 静止膜電位 [mV]
    vreset::FT = -65 # リセット電位 [mV]
    vthr::FT = -40 # 閾値電位 [mV]
    vpeak::FT = 30 #ピーク電位 [mV]
end

# LIFニューロンの定義
@kwdef mutable struct LIF{FT} <: AbstractNeuron{FT}
    param::LIFParameter = LIFParameter{FT}()
    N::UInt32 #ニューロンの数
    v::Vector{FT} = fill(-65.0, N); v_::Vector{FT} = fill(-65.0, N) # 膜電位, 発火電位も記録する膜電位 (mV)
    spike::Vector{Bool} = zeros(Bool, N) # 発火
    tlast::Vector{FT} = zeros(N) # 最後の発火時刻 [ms]
    tcount::FT = 0 # 時間カウント
end

# LIFNeuronに対するupdate!メソッドの定義
function update!(neurons::LIF{FT}, param::LIFParameter{FT}, dt::FT, Ie::Vector{FT}) where FT
    @unpack N, v, v_, spike, tlast, tcount = neurons
    @unpack tref, tau_m, vrest, vreset, vthr, vpeak = param
    
    @inbounds for i = 1:N
        v[i] += dt * ((dt*tcount) > (tlast[i] + tref))*((vrest - v[i] + Ie[i]) / tau_m)
    end
    
    @inbounds for i = 1:N
        spike[i] = v[i] >= vthr
        v_[i] = ifelse(spike[i], vpeak, v[i]) #発火時の電位も含めて記録するための変数
        v[i] = ifelse(spike[i], vreset, v[i])        
        tlast[i] = ifelse(spike[i], dt*tcount, tlast[i]) # 発火時刻の更新
    end
    # time count +1
    neurons.tcount += 1
end

#= update関数の検証について
    N=1 [個], dt=0.01f0 [ms], T=450 [ms]の環境下で
    paramを引数として明示的に渡すと実行時間は約0.002
    paramをLIFから参照すると実行時間は約0.02
    paramをmutableではない構造体として渡す方が10倍ほど早い
=#

# LIFNeuronに対するinit!メソッドの定義
function init!(neurons::LIF)
    @unpack N, v, v_, spike, tlast, tcount = neurons
    v = zeros(N)
    v_ = zeros(N)
    tlast = zeros(N)
    spike = zeros(N)
    tcount = 0
end

function get_spike(neurons::LIF)::Vector{Bool}
    return neurons.spike
end

function get_v(neurons::LIF{FT})::Vector{FT} where FT
    return neurons.v
end

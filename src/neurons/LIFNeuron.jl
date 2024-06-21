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
@kwdef mutable struct LIF{FT,UIT} <: AbstractNeuron{FT}
    param::LIFParameter{FT} = LIFParameter{FT}()
    N::UIT #ニューロンの数
    v::Vector{FT} = fill(-65.0, N); v_::Vector{FT} = fill(-65.0, N) # 膜電位, 発火電位も記録する膜電位 (mV)
    spike::BitVector = BitVector(zeros(Bool, N))
    tlast::Vector{FT} = zeros(N) # 最後の発火時刻 [ms]
    tcount::UIT = 0 # 時間カウント
end

# LIFNeuronに対するupdate!メソッドの定義
function update!(neurons::LIF{FT,UIT}, param::LIFParameter{FT}, dt::FT, Ie::SubArray) where {FT,UIT}
    @unpack N, v, v_, spike, tlast, tcount = neurons
    @unpack tref, tau_m, vrest, vreset, vthr, vpeak = param
    
    @inbounds for i = 1:N
        vtmp = v[i]
        tlasttmp = tlast[i]
        
        vtmp += dt * ((dt*tcount) > (tlasttmp + tref))*((vrest - vtmp + Ie[i]) / tau_m)
        spiketmp = vtmp >= vthr

        spike[i] = spiketmp
        v_[i] = ifelse(spiketmp, vpeak, vtmp) #発火時の電位も含めて記録するための変数
        v[i] = ifelse(spiketmp, vreset, vtmp)        
        tlast[i] = ifelse(spiketmp, dt*tcount, tlasttmp) # 発火時刻の更新
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

# LIFNeuronに対するupdate!メソッドの定義
function update_threads!(neurons::LIF{FT,UIT}, param::LIFParameter{FT}, dt::FT, Ie::SubArray) where {FT,UIT}
    @unpack N, v, v_, spike, tlast, tcount = neurons
    @unpack tref, tau_m, vrest, vreset, vthr, vpeak = param
    
    Threads.@threads for i = 1:N
        @inbounds begin
            vtmp = v[i]
            tlasttmp = tlast[i]
            
            vtmp += dt * ((dt*tcount) > (tlasttmp + tref))*((vrest - vtmp + Ie[i]) / tau_m)
            spiketmp = vtmp >= vthr
    
            spike[i] = spiketmp
            v_[i] = ifelse(spiketmp, vpeak, vtmp) #発火時の電位も含めて記録するための変数
            v[i] = ifelse(spiketmp, vreset, vtmp)        
            tlast[i] = ifelse(spiketmp, dt*tcount, tlasttmp) # 発火時刻の更新
        end
    end
    # time count +1
    neurons.tcount += 1
end

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

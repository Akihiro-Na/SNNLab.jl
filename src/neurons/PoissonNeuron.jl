#= 
Poisson point process neuron の定義ファイル
環境からの入力用
ポアソン点過程では時刻0からtまでにn回スパイクが生じる確率がポアソン分布P[N(t)=n] = (λt)ⁿ/n! × exp(-λt)に従う
ここでλは強度と呼ばれ，発火しやすさ（1秒間に何回発火するか，つまり周波数）を表す．
別の定義として，ポアソン点過程はその区間で一度しか発火しないほどごく短い時間区間[t, t+Δt]の間に
スパイクが生じる確率がP[N(t+Δt)-N(t)=1] = λ(t)Δt + o(Δt)に従う．
ポアソン点過程ではスパイク発生間隔(Inter spike interval, ISI)はf(t,λ) = λexp(-λt), (t≥0)に従う．
=#


# Poisson point process neuron model のパラメータ(固定)
#=
@kwdef struct PPPNeuronParameter{FT} <: AbstractNeuronParam{FT}
end
=#

# Poisson point process neuron modelの定義
@kwdef mutable struct PPPNeuron{FT,UIT} <: AbstractNeuron{FT}
    N::UIT #ニューロンの数
    nt::UIT #用意する時間ステップ数
    tcount::UIT = 1 # 時間カウント
    random_numbers::Matrix{FT} = rand(nt, N)
    spike::BitVector = BitVector(zeros(Bool, N))
end

# Poisson point process neuron modelのupdate!メソッドの定義
function update!(neurons::PPPNeuron{FT,UIT}, dt::FT, λ) where {FT,UIT}
    @unpack random_numbers, tcount, spike = neurons

    # 乱数の配列が使い切られた場合、init! 関数を使用して新しい乱数を生成する
    if tcount > size(random_numbers, 1)
        init!(neurons)
        @unpack random_numbers, tcount, spike = neurons
    end
    
    # スパイクが生じるかの判定
    @inbounds for i in eachindex(spike)
        spike[i] = random_numbers[tcount, i] < λ[i] * dt * 1e-3
    end
    # time count +1
    neurons.tcount += 1

    return neurons.spike
end

# LIFNeuronに対するinit!メソッドの定義
function init!(neurons::PPPNeuron{FT}) where FT
    @unpack N, nt = neurons
    neurons.tcount = 1 # 時間カウント
    rand!(neurons.random_numbers)
    fill!(neurons.spike, false)
end

function get_spike(neurons::PPPNeuron)::BitVector
    return neurons.spike
end


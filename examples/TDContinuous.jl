#=
TDContinuousのtest
報酬に対する応答の確認
=#
using SNNLab
using Plots

#function run_TDtest()


    FT = Float32
    UIT = UInt32
    Ncritic::UIT = 2
    td = TDContinuous{FT,UIT}(N=Ncritic)
    dt::FT = 1 # ms
    spike::BitVector = [1, 0]
    reward::FT = 100.0
    update!(td, td.param, dt, spike, reward)

    

#end

#run_TDtest()
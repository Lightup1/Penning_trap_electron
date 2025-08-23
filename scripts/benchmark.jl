using BenchmarkTools
using QuantumToolbox
##
σ=sigmax()
Ω=1
nh=4e-2
N = 30
ψ0 = basis(2,1) ⊗ basis(2,1) ⊗ fock(N,0)
N = length(ψ0) ÷ 4  # assuming ψ0 is a vector of length 4*N

σ1x=σ⊗qeye(2)⊗qeye(N)
σ2x=qeye(2)⊗σ⊗qeye(N)
a=qeye(2)⊗qeye(2)⊗destroy(N)

c_ops = [sqrt(nh) * a,
            sqrt(nh) * a']

Δ=Ω*4

ϕ=2pi*Ω^2/Δ^2
U_ideal = exp(1im*ϕ*(σ1x+σ2x)^2)

ψend = U_ideal * ψ0

ψend_p=ptrace(ψend,(1,2))

H=Ω*(σ1x+σ2x)*(a+a')+Δ*a'*a

tlist = [0,2*2pi/Δ] # a list of time points of interest
# eop_ls = [
#     ψend_p⊗qeye(N),              # ideal end state
# ];
##
sol = mesolve(H, ψ0, tlist, c_ops)
# @benchmark sol = mesolve($H, $ψ0, $tlist, $c_ops; progress_bar=$Val(false))
# 1-real(sol.expect[1, end])  # infidelity
using BenchmarkTools
using QuantumOptics
using CUDA
# CUDA.allowscalar(true)
function to_gpu(a::Operator)
    data = cu(a.data)
    gpu_operator = Operator(a.basis_l, a.basis_r, data)
    return gpu_operator
end

function to_gpu(a::Ket)
    data = cu(a.data)
    gpu_operator = Ket(a.basis, data)
    return gpu_operator
end
##
N = 30
b1 = SpinBasis(1//2)
b2=SpinBasis(1//2)
b3= FockBasis(N)
σ1x=sigmax(b1) ⊗ identityoperator(b2) ⊗ identityoperator(b3)
σ2x=identityoperator(b1) ⊗ sigmax(b2) ⊗ identityoperator(b3)
a=identityoperator(b1) ⊗ identityoperator(b2) ⊗ destroy(b3)
Ω=1
nh=4e-2
ψ0 = spinup(b1) ⊗ spinup(b2) ⊗ fockstate(b3,0)
N = length(ψ0) ÷ 4  # assuming ψ0 is a vector of length 4*N

##

c_ops = [sqrt(nh) * a,
            sqrt(nh) * a']

Δ=Ω*4

ϕ=2pi*Ω^2/Δ^2
U_ideal = exp(1im*ϕ*(σ1x+σ2x)^2)

ψend = U_ideal * ψ0

ψend_p=ptrace(ψend,(3,))
eop_ls = [
    ψend_p⊗identityoperator(b3),              # ideal end state
];

H=Ω*(σ1x+σ2x)*(a+a')+Δ*a'*a
##
# H=cu(H)
ψ0_gpu=to_gpu(ψ0);

##
tlist = [0,1*2pi/Δ] # a list of time points of interest
fout_func=eop_ls-> ((t,psi)->[expect(eop,psi) for eop in eop_ls])
fout = fout_func(eop_ls)
##
sol=timeevolution.master(tlist, ψ0, H, c_ops)
# sol = mesolve(H, ψ0, tlist, c_ops; e_ops = eop_ls)
# @benchmark sol = mesolve($H, $ψ0, $tlist, $c_ops; progress_bar=$Val(false))
@benchmark sol=timeevolution.master($tlist, $ψ0, $H, $c_ops)
@benchmark sol=timeevolution.master($tlist, $ψ0_gpu, $to_gpu(H), $to_gpu.(c_ops))
# 1-real(sol.expect[1, end])  # infidelity
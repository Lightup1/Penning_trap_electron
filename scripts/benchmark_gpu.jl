using BenchmarkTools
using QuantumToolbox
using CUDA
using DiffEqGPU
CUDA.allowscalar(false)
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

eop_ls = [
    ψend_p⊗qeye(N),              # ideal end state
];

H=Ω*(σ1x+σ2x)*(a+a')+Δ*a'*a
##
H=cu(H)
ψ0=cu(ψ0)
c_ops=cu.(c_ops)
eop_ls=cu.(eop_ls)

##
tlist = [0,4*2pi/Δ] # a list of time points of interest

##
sol = mesolve(H, ψ0, tlist, c_ops; e_ops = eop_ls)
@benchmark sol = mesolve($H, $ψ0, $tlist, $c_ops; progress_bar=$Val(false))
CUDA.@time sol = mesolve(H, ψ0, tlist, c_ops; progress_bar=Val(false));
# 1-real(sol.expect[1, end])  # infidelity
##
using QuantumToolbox
using CUDA
CUDA.allowscalar(false) # Avoid unexpected scalar indexing
using BenchmarkTools 
##
N = 200 # cutoff of the Hilbert space dimension
ω = 1.0 # frequency of the harmonic oscillator
γ = 0.1 # damping rate

a_gpu = cu(destroy(N)) # The only difference in the code is the cu() function

H_gpu = ω * a_gpu' * a_gpu

ψ0_gpu = cu(fock(N, 20))


c_ops = [sqrt(γ) * a_gpu]
e_ops = [a_gpu' * a_gpu]

tlist = [0,100] # time list
sol = mesolve(H_gpu, ψ0_gpu, tlist, c_ops, e_ops = e_ops)

##
@benchmark mesolve($H_gpu, $ψ0_gpu, $tlist, $c_ops, e_ops = $e_ops,progress_bar=$Val(false))
##
ρ0_gpu = ψ0_gpu*ψ0_gpu'
dense_0=QuantumToolbox.to_dense(QuantumToolbox._complex_float_type(ComplexF64), QuantumToolbox.mat2vec(ket2dm(ψ0_gpu).data))
L_evo = QuantumToolbox._mesolve_make_L_QobjEvo(H_gpu, c_ops)
L=L_evo.data
T = Base.promote_eltype(L_evo, ψ0_gpu)
buffer=L*(ψ0_gpu'*ψ0_gpu) 
prob=QuantumToolbox.ODEProblem{QuantumToolbox.getVal(Val(true)),QuantumToolbox.FullSpecialize}(L, QuantumToolbox.to_dense(QuantumToolbox._complex_float_type(ComplexF64), copy(ψ0_gpu.data)), tlist, [1])
##
@benchmark QuantumToolbox.solve($prob,$QuantumToolbox.Tsit5())
##
@benchmark begin
    for i in 1:10
        f(w, du,u,p,t) 
    end
end setup=(f=prob.f;w=copy(dense_0); du=copy(dense_0);u=copy(dense_0);p=[1];t=0)
##
prob.f(copy(dense_0),copy(dense_0),copy(dense_0),[1],0) 
##
using LinearAlgebra
using SparseArrays
using BenchmarkTools
using CUDA
CUDA.allowscalar(false)
##
# Dense case works
A = CUDA.rand(ComplexF64, 1000, 1000)
x = CUDA.rand(ComplexF64, 1000)
y = similar(x)

@btime mul!($y, $A, $x);
@code_warntype mul!(y, A, x)
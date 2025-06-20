"""
δ is the static motional frequency shift.

Ω is the spin motion coupling coefficient in formula: 2ħΩf(t)S_α(a†+a)

na is the phonon number

ψ is the initial qubit state

S is the multiqubit Pauli spin operator 

N represents the "N" of the N-loop gate 

"""
function infidelity_static_motional_shift(δ,Ω,na,ψ,S,N)
    λ2S = variance(S,ψ) 
    λ2S2 = variance(S^2,ψ) 
    π^2*δ^2/(64*Ω^2)*((2na+1)*λ2S+λ2S2/4N)
end

"""
ϵ describes the quatic anharmonicity.

Ω is the spin motion coupling coefficient in formula: 2ħΩf(t)S_α(a†+a)

na is the phonon number

ψ is the initial qubit state

S is the multiqubit Pauli spin operator 

N represents the "N" of the N-loop gate 

"""
function infidelity_trap_anharmonicity(ϵ,Ω,na,ψ,S,N)

    λ2S = variance(S,ψ) 
    λ2S2 = variance(S^2,ψ) 
    (9 * π^2 * ϵ^2) / (16 * Ω^2) * (
    λ2S * (
        4 * (2*na^3 + 3 * na^2 + 3 * na + 1) +
        (6 / N) * (2*na^2 + 2 * na + 1) +
        (9 / (4 * N^2)) * (2 * na + 1)
    ) +
    λ2S2 * (
        (3 / (8 * N)) * (11 * na^2 + 11 * na + 3) +
        (3 / (4 * N^2)) * (2 * na + 1) +
        (9 / (64 * N^3))
    )
    )
end


"""
Ωpp describes the second-order field inhomogeneity.

Ω is the spin motion coupling coefficient in formula: 2ħΩf(t)S_α(a†+a)

na is the phonon number

ψ is the initial qubit state

S is the multiqubit Pauli spin operator 

N represents the "N" of the N-loop gate 

"""
function infidelity_field_inhomogeneity(Ωpp,Ω,na,ψ,S,N)
    λ2S = variance(S,ψ) 
    λ2S2 = variance(S^2,ψ) 
    
    (9π^2 * Ωpp^2) / (16 * Ω^2) * (
    λ2S2 * (
        4 * na^2 + 4 * na + 1 +
        (3 / (2N)) * (2 * na + 1) +
        9 / (16 * N^2)
    ) + 
    (4 / N) * λ2S * (2 * na + 1)
    )
end


"""
nh is the heating rate.

Ω is the spin motion coupling coefficient in formula: 2ħΩf(t)S_α(a†+a)

na is the phonon number

ψ is the initial qubit state

S is the multiqubit Pauli spin operator 

N represents the "N" of the N-loop gate 

"""
function infidelity_heating(nh,Ω,na,ψ,S,N)
    λ2S = variance(S,ψ) 
    π*nh*λ2S/(8*Ω*sqrt(N))
end

"""
η describes the decay rate with 2/η = τ (coherence time).

Ω is the spin motion coupling coefficient in formula: 2ħΩf(t)S_α(a†+a)

na is the phonon number

ψ is the initial qubit state

S is the multiqubit Pauli spin operator 

N represents the "N" of the N-loop gate 

"""
function infidelity_motional_dephasing(η,Ω,na,ψ,S,N)
    λ2S = variance(S,ψ) 
    λ2S2 = variance(S^2,ψ) 
    (π * η) / (16 * Ω * sqrt(N)) * (
    (2 * na + 1) * λ2S + (3 / (16 * N)) * λ2S2
    )
end

"""
Calculate infidelity resulting from a static motional frequency shift.

Ω is the two-qubit gate Rabi frequency,

δ is the detuning of the motional frequency shift from the ideal value,

ψ0 is the initial state of the system,

The function returns the infidelity of the two-qubit gate.
"""
function infidelity_static_motional_shift_numerical(Ω,δ,ψ0;kwargs...)
    # use ψ0 to get the cutoff number of motional states, N
    N = length(ψ0) ÷ 4  # assuming ψ0 is a vector of length 4*N

    σ1x=sigmax()⊗qeye(2)⊗qeye(N)
    σ2x=qeye(2)⊗sigmax()⊗qeye(N)
    a=qeye(2)⊗qeye(2)⊗destroy(N)

    Δ=Ω*4

    ϕ=2pi*Ω^2/Δ^2
    U_ideal = exp(1im*ϕ*(σ1x+σ2x)^2)

    ψend = U_ideal * ψ0
    ψend_p=ptrace(ψend,(1,2))

    H=Ω*(σ1x+σ2x)*(a+a')+(δ+Δ)*a'*a

    tlist = [0,2pi/Δ] # a list of time points of interest
    eop_ls = [
        ψend_p⊗qeye(N),               # ideal end state
    ];

    sol = sesolve(H , ψ0, tlist; e_ops = eop_ls,kwargs...)
    1-real(sol.expect[1, end])  # infidelity
end

"""
Infidelity resulting from an anharmonicity in the trap potential.

Ω is the two-qubit Rabi frequency, 

ϵ is the anharmonicity strength,

ψ0 is the initial state.

σ is the spin operator, default is sigmax().
"""
function infidelity_trap_anharmonicity_numerical(Ω,ϵ,ψ0;σ=sigmax(),kwargs...)
    N = length(ψ0) ÷ 4  # assuming ψ0 is a vector of length 4*N

    σ1x=σ⊗qeye(2)⊗qeye(N)
    σ2x=qeye(2)⊗σ⊗qeye(N)
    a=qeye(2)⊗qeye(2)⊗destroy(N)

    Δ=Ω*4

    ϕ=2pi*Ω^2/Δ^2
    U_ideal = exp(1im*ϕ*(σ1x+σ2x)^2)
    ψend = U_ideal * ψ0
    ψend_p=ptrace(ψend,(1,2))

    H=Ω*(σ1x+σ2x)*(a+a')+Δ*a'*a+6ϵ*(a'*a+(a'*a)^2)


    tlist = [0,2pi/Δ] # a list of time points of interest
    eop_ls = [
        ψend_p⊗qeye(N),              # ideal end state
    ];

    sol = sesolve(H , ψ0, tlist; e_ops = eop_ls,kwargs...)
    1-real(sol.expect[1, end])  # infidelity
end

"""
Infidelity resulting from an anharmonicity in the trap potential.

Ω is the two-qubit Rabi frequency, 

Ω2 is the second-order field inhomogeneity strength,

ψ0 is the initial state.

σ is the spin operator, default is sigmax().
"""
function infidelity_field_inhomogeneity_numerical(Ω,Ω2,ψ0;σ=sigmax(),kwargs...)
    N = length(ψ0) ÷ 4  # assuming ψ0 is a vector of length 4*N

    σ1x=σ⊗qeye(2)⊗qeye(N)
    σ2x=qeye(2)⊗σ⊗qeye(N)
    a=qeye(2)⊗qeye(2)⊗destroy(N)

    Δ=Ω*4

    ϕ=2pi*Ω^2/Δ^2
    U_ideal = exp(1im*ϕ*(σ1x+σ2x)^2)
    ψend = U_ideal * ψ0
    ψend_p=ptrace(ψend,(1,2))

    H=Ω*(σ1x+σ2x)*(a+a')+Δ*a'*a+3Ω2*(σ1x+σ2x)*(a'*a*a'+a*a'*a)

    tlist = [0,2pi/Δ] # a list of time points of interest
    eop_ls = [
        ψend_p⊗qeye(N),              # ideal end state
    ];

    sol = sesolve(H , ψ0, tlist; e_ops = eop_ls,kwargs...)
    1-real(sol.expect[1, end])  # infidelity
end



"""
Infidelity resulting from an anharmonicity in the trap potential.

Ω is the two-qubit Rabi frequency, 

η is the dephasing rate,

ψ0 is the initial state.

σ is the spin operator, default is sigmax().
"""
function infidelity_motional_dephasing_numerical(Ω,η,ψ0::QuantumObject{Ket};σ=sigmax(),kwargs...)
    N = length(ψ0) ÷ 4  # assuming ψ0 is a vector of length 4*N

    σ1x=σ⊗qeye(2)⊗qeye(N)
    σ2x=qeye(2)⊗σ⊗qeye(N)
    a=qeye(2)⊗qeye(2)⊗destroy(N)

    c_ops = [sqrt(η) * a'*a]

    Δ=Ω*4

    ϕ=2pi*Ω^2/Δ^2
    U_ideal = exp(1im*ϕ*(σ1x+σ2x)^2)

    ψend = U_ideal * ψ0

    ψend_p=ptrace(ψend,(1,2))

    H=Ω*(σ1x+σ2x)*(a+a')+Δ*a'*a

    tlist = [0,2pi/Δ] # a list of time points of interest
    eop_ls = [
        ψend_p⊗qeye(N),              # ideal end state
    ];

    sol = mesolve(H , ψ0, tlist, c_ops; e_ops = eop_ls,kwargs...)
    1-real(sol.expect[1, end])  # infidelity
end


"""
Infidelity resulting from an anharmonicity in the trap potential.

Ω is the two-qubit Rabi frequency, 

nh is the heating rate,

ψ0 is the initial state.

σ is the spin operator, default is sigmax().
"""
function infidelity_heating_numerical(Ω,nh,ψ0::QuantumObject{Ket};σ=sigmax(),kwargs...)
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

    tlist = [0,2pi/Δ] # a list of time points of interest
    eop_ls = [
        ψend_p⊗qeye(N),              # ideal end state
    ];

    sol = mesolve(H, ψ0, tlist, c_ops; e_ops = eop_ls,kwargs...)
    1-real(sol.expect[1, end])  # infidelity
end
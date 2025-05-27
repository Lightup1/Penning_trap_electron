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
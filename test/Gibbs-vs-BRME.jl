
using AutoDiffOQS, Zygote, FiniteDifferences, DifferentialEquations, DiffEqSensitivity#, Plots

# Check that BRME & PME with only phonon processes relaxes to the correct Gibbs state and that auto-diff gradients agree

dE = 0.01
J = 0.001
T = 300

Hamiltonian(dE, J) = dE * sigma_z + J * sigma_x

const kb_eV = AutoDiffOQS.kb / AutoDiffOQS.e
nbe(w, T) = ( exp(abs(w) / (kb_eV * T)) - 1 )^-1.0
Sw(w, T) =  w == 0 ? 0 : (nbe(w, T) + (w > 0))
# Sw(w, T) =  w == 0 ? 0 : 1e-2*abs(w)^3*exp(-(abs(w)/0.5)^2)*(nbe(w, T) + (w > 0))
a_ops = [[projection(2, i, i), w -> Sw(w, T)] for i in 1:2]
ρ0 = projection(2, 1, 1)

ss_gibbs, grad_gibbs = withjacobian((dE, J) -> gibbs_dist(Hamiltonian(dE, J), T), dE, J)
ss_brme, grad_brme = withjacobian((dE, J) -> bloch_redfield_steady_state(Hamiltonian(dE, J), a_ops, ρ0; use_secular=false) |> diag |> real, dE, J)
ss_pauli, grad_pauli = withjacobian((dE, J) -> pauli_steady_state(Hamiltonian(dE, J), a_ops, ρ0) |> diag |> real, dE, J)

println("Gibbs state populations:\t\t\t", ss_gibbs)
println("Pauli steady state populations: \t\t", ss_pauli)
println("Bloch-Redfield steady state populations: \t", ss_brme)
println()
println("Gibbs derivatives: \t\t\t\t", grad_gibbs)
println("Pauli derivatives: \t\t\t\t", grad_pauli)
println("Bloch-Redfield derivatives: \t\t\t", grad_brme)


#Sanity checks using finite difference
central_fdm(5, 1)(x -> gibbs_dist(Hamiltonian(x, J), T), dE) |> println
central_fdm(5, 1)(x -> bloch_redfield_steady_state(Hamiltonian(x, J), a_ops, ρ0) |> diag |> real, dE) |> println
central_fdm(5, 1)(x -> pauli_steady_state(Hamiltonian(x, J), a_ops, ρ0) |> diag |> real, dE) |> println


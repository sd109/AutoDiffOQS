
using AutoDiffOQS, Zygote

# Check that a BRME with only phonon processes relaxes to the correct Gibbs state and that auto-diff gradients agree

dE = 0.01
J = 0.001
T = 10

H(dE, J) = dE * sigma_z + J * sigma_x

kb_eV = AutoDiffOQS.kb / AutoDiffOQS.e
nbe(w, T) = ( exp(abs(w) / (kb_eV * T)) - 1 )^-1.0
Sw(w, T) =  w == 0 ? 0 : (nbe(w, T) + (w > 0))
a_ops = [[projection(2, i, i), w -> Sw(w, T)] for i in 1:2]
ρ0 = projection(2, 1, 1)

ss_brme, grad_brme = withjacobian((dE, J) -> bloch_redfield_steady_state(complex(H(dE, J)), a_ops, ρ0; use_secular=false) |> diag |> real, dE, J)
ss_gibbs, grad_gibbs = withjacobian((dE, J) -> gibbs_dist(H(dE, J), T), dE, J)

println("Bloch-Redfield steady state populations: \t", ss_brme)
println("Gibbs state populations:\t\t\t", ss_gibbs)

println("\nBloch-Redfield derivatives: \t\t\t", grad_brme)
println("Gibbs derivatives: \t\t\t\t", grad_gibbs)



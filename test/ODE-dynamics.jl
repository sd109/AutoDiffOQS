
using Pkg; Pkg.activate("/home/scottd/Dropbox/Physics/Research Work/Julia/MBAM-v2/")
using AutoDiffOQS, Zygote, DifferentialEquations, DiffEqSensitivity, Plots

Hamiltonian(dE, J) = dE*sigma_z + J*sigma_x

function model(dE, J)

    H = Hamiltonian(dE, J)
    ρ0 = projection(2, 1, 1)
    c_ops = [1e-1*sigma_z]
    L = liouvillian(H, c_ops)

    times = range(0, 100, length=100)
    sol = ode_dynamics(L, ρ0, times; save_idxs=diagind(ρ0))
    populations = real(sol[:, :])
    return populations
end

# Ps = model(0.0, 0.1)
# plot(Ps')

val, grad = withjacobian(model, 0.01, 0.1)
Ps = reshape(val, 2, :)
dE_grad = reshape(grad[1], 2, :)
J_grad = reshape(grad[2], 2, :)

plot(
    plot(Ps', title="Populations"),
    plot(dE_grad', title="dP/d(dE)"),
    plot(J_grad', title="dP/dJ"),
    label=["P1" "P2"],
    layout=(3, 1),
    size=(800, 800),
)
savefig("ODE-dynamics-test.png")
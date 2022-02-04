

using AutoDiffOQS, Zygote, DifferentialEquations, DiffEqSensitivity, Plots

# Hamiltonian(dE, J) = dE*sigma_z + J*sigma_x
Hamiltonian(p) = p[1]*sigma_z + p[2]*sigma_x

ρ0 = projection(2, 1, 1) |> complex
c_ops = [1e-4*sigma_z]

L = liouvillian(Hamiltonian(0.0, 0.1), c_ops)

f(u, p, t) = L * u
tspan = (0.0, 100.0)
prob = ODEProblem(f, reshape(ρ0, :), tspan)
sol = solve(prob)

times = range(tspan..., length=100)
plot(times, real(sol(times))[1, :])


function diff_ode(dE, J)

    # H = Hamiltonian(dE, J)
    # ρ0 = projection(2, 1, 1) |> complex
    # c_ops = [1e-2*sigma_z]
    # L = liouvillian(H, c_ops)
    # f(u, p, t) = L * u

    f(u, p, t) = liouvillian(Hamiltonian(p), c_ops) * u

    times = (0.0, 100.0)
    prob = ODEProblem(f, reshape(ρ0, :), times, [dE, J])
    sol = solve(prob)

    return real(sol.u[end])
end

diff_ode(0.0, 0.1)
    
withjacobian(diff_ode, 0.0, 0.1)




function ode_dynamics(L, ρ0, times; kwargs...)
    prob = ODEProblem((u, p, t) -> p * u, complex(reshape(ρ0, :)), extrema(times), L) #ODE 'parameters' are just liouvillian L
    return solve(prob; saveat=times, kwargs...)
end 

function model(dE, J)

    H = Hamiltonian(dE, J)
    ρ0 = projection(2, 1, 1)
    c_ops = [1e-1*sigma_z]
    L = liouvillian(H, c_ops)

    times = range(0, 100, length=100)
    sol = ode_dynamics(L, ρ0, times)

    states = map(ρ -> reshape(ρ, size(H)...), sol.u)
    populations = reduce(hcat, real(diag.(states)))

    return populations
end

Ps = model(0.0, 0.1)
plot(Ps')

val, grad = withjacobian(model, 0.01, 0.1)

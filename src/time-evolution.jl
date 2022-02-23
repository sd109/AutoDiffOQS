



# -------------------------------------------------------------------------------------------------------------------- #
#                                           Generators (Lindblad, BRME & PME)                                          #
# -------------------------------------------------------------------------------------------------------------------- #



# ----------------------------------------------------- Lindblad ----------------------------------------------------- #

# Inspired by QuantumOpticsBase.jl's liouvillian implementation but adapted to work with Zygote auto-diff

spost(op::AbstractMatrix) = kron(permutedims(op), I(size(op, 1)))
spre(op::AbstractMatrix) = kron(I(size(op, 1)), op)


function liouvillian(H, c_ops)
    L = spre(-1im*H) + spost(1im*H)
    for op in c_ops
        op_dagger = op'
        op_dagger_op = 0.5*op_dagger*op
        L -= spre(op_dagger_op) + spost(op_dagger_op)
        L += spre(op) * spost(op_dagger)
    end
    return L
end


# ------------------------------------------------------- BRME ------------------------------------------------------- #

function bloch_redfield_tensor(H::AbstractMatrix, a_ops::Array; c_ops=[], use_secular=true, secular_cutoff=0.1)
    
    #Check that a_ops are all Hermitian
    herm_check = is_herm.(getindex.(a_ops, 1))
    all(herm_check) || error("All 'A' operators must be Hermitian\nNon-herm ops: $(.!(herm_check))")

    H = Hermitian(complex(H))  # H must be complex so that ChainRules.eigen_rev rule works correctly
    # Use the energy eigenbasis
    H_evals, transf_mat = eigen(H)  

    N = length(H_evals) #Hilbert space dimension
    K = length(a_ops) #Number of system-env interation operators

    # Calculate Liouvillian for Lindblad terms (this also includes unitary dynamics part dρ/dt = -i[H, ρ])
    Heb = to_Heb(H, transf_mat)
    L = liouvillian(Heb, [to_Heb(op, transf_mat) for op in c_ops])
    
    #If only Lindblad collapse terms (no a_ops given) then we're done
    if K==0
        return L, transf_mat #Liouvillian is in the energy eigenbasis here
    end

    #Transform interaction operators to Hamiltonian eigenbasis
    A = Array{ComplexF64}(undef, N, N, K) |> Zygote.Buffer
    for k in 1:K
        A[:, :, k] = to_Heb(a_ops[k][1], transf_mat)
    end

    # Array of transition frequencies between eigenstates
    W = H_evals .- transpose(H_evals)

    #Array for spectral functions evaluated at transition frequencies
    Jw = Array{Float64}(undef, N, N, K) |> Zygote.Buffer
    #Loop over all a_ops and calculate each spectral density at all transition frequencies
    for k in 1:K
        Jw[:, :, k] = a_ops[k][2].(W)
    end

    #Calculate secular cutoff scale if needed
    if use_secular
        dw_min = minimum(abs.(W[W .!= 0.0]))
        w_cutoff = dw_min * secular_cutoff
    end

    #Initialize R_abcd array
    data = zeros(ComplexF64, N, N, N, N)
    idxs = CartesianIndices(data)
    data = Zygote.bufferfrom(data)
    #Loop through all indices and calculate elements - seems to be as efficient as any fancy broadcasting implementation (and much simpler to read)
    for idx in idxs

        a, b, c, d = Tuple(idx) #Unpack indices

        #Skip any values that are larger than the secular cutoff
        if use_secular && abs(W[a, b] - W[c, d]) > w_cutoff
            continue
        end

        """ Term 1 """
        data[a, b, c, d] = sum(A[a, c, :] .* A[d, b, :] .* (Jw[c, a, :] .+ Jw[d, b, :])) #Broadcasting over interaction operators

        """ Term 2 (b == d) """
        if b == d
            data[a, b, c, d] -= sum( A[a, :, :] .* A[:, c, :] .* Jw[c, :, :] ) #Broadcasting over interaction operators and extra sum over n
        end

        """ Term 3 (a == c) """
        if a == c
            data[a, b, c, d] -= sum( A[d, :, :] .* A[:, b, :] .* Jw[d, :, :] ) #Broadcasting over interaction operators and extra sum over n
        end

    end
    
    data = copy(data) #Convert back from Zygote.Buffer type once we're finished indexing
    data *= 0.5 #Don't forget the factor of 1/2
    data = reshape(data, N^2, N^2) #Convert to Liouville space
    R = sparse(data) #Remove any zero values and convert to sparse array

    #Add Bloch-Redfield part to unitary dynamics and Lindblad Liouvillian calculated above
    return L+R, transf_mat

end #Function


# -------------------------------------------------------- PME ------------------------------------------------------- #

function pauli_generator(H, a_ops)

    #Check that a_ops are all Hermitian
    herm_check = is_herm.(getindex.(a_ops, 1))
    all(herm_check) || error("All 'A' operators must be Hermitian\nNon-herm ops: $(.!(herm_check))")


    N = size(H, 1)
    K = length(a_ops)

    #Make complex Hermitian explicitly so that ChainRules.eigen_rev works properly
	H = Hermitian(complex(H))
    # Get eigenenergy differences
    evals, transf_mat = eigen(H)
    inv_transf_mat = inv(transf_mat)
    diffs = evals' .- evals #Matrix of eigenenergy differences

    #Pre-allocate output matrix
    W_matrix = zeros(N, N)
    # W_matrix = Zygote.bufferfrom(zeros(N, N))
    for i in 1:K #Loop through a_ops    
        A_eb = inv_transf_mat * a_ops[i][1] * transf_mat
		W_matrix += real(a_ops[i][2].(diffs) .* A_eb .* A_eb') #Can we enforce real here? I think so since we're doing |<x|A|y>|^2
	end

    #Add additional required term to each diagonal element of L
    # L = Zygote.bufferfrom(W_matrix)
    # for i in 1:N
    #     L[i, i] -= sum(L[:, i])
    # end
    #Buffer version above doesn't work properly (result is right but gradients are wrong with no error or warning...)
    #Use this (less efficient) method for now
    L = W_matrix - diagm(sum.(eachcol(W_matrix)))

    return L, transf_mat
end



# -------------------------------------------------------------------------------------------------------------------- #
#                                                     Steady states                                                    #
# -------------------------------------------------------------------------------------------------------------------- #


#Can't use nullspace method here because nullspace(::ComplexMatrix) internally uses non-julia functions for svd calc (unlike nullspace(::RealMatrix) case)
function steady_state(L, ρ0; tol=1e-15)
    ρ0_vec = reshape(ρ0, :)
    vals, vecs = eigen(L)
    idxs = findall(abs.(vals) .< tol)
    length(idxs) == 0 && throw(error("Eigenvalues of L are all > $(tol). Are you sure the system reaches a steady state?"))
    ss_vec = sum(vecs[:, i] * vecs[:, i]' * ρ0_vec for i in idxs)
    ss = Array(reshape(ss_vec, size(ρ0)))
    return ss /= tr(ss)
end


function bloch_redfield_steady_state(H, a_ops, ρ0; kwargs...)
    R, U = bloch_redfield_tensor(H, a_ops; kwargs...)
    ρ0_eb = inv(U) * ρ0 * U #Transform to eigenbasis
    ss_eb = steady_state(R, ρ0_eb)
    return U * ss_eb * inv(U) #Transform back from eigenbasis
end


function pauli_steady_state(H, a_ops, ρ0)
    W, U = pauli_generator(H, a_ops)
    ρ0_eb = to_Heb(ρ0, U) #inv(U) * ρ0 * U
    P0 = real(diag(ρ0_eb))
    vals, vecs = eigen(W)
    idxs = findall(abs.(vals) .< 1e-15)
    P_ss = sum(vecs[:, i] * vecs[:, i]' * P0 for i in idxs)
    ss_eb = diagm(P_ss) / sum(P_ss)
    return U * ss_eb * inv(U)
end


function gibbs_state(H, T)
    A = exp(-H/(kb_eV*T))
    B = tr(A)
    return A/B
end

gibbs_dist(H, T) = diag(gibbs_state(H, T))



# -------------------------------------------------------------------------------------------------------------------- #
#                                                       Dynamics                                                       #
# -------------------------------------------------------------------------------------------------------------------- #


function expLt(H, c_ops, t, ρ0)
    L = liouvillian(H, c_ops)
    ρt_vec = exp(L*t)*reshape(ρ0, :)
    ρt = reshape(ρt_vec, size(ρ0)...)
    return ρt
end

function ode_dynamics(L, ρ0, times; kwargs...)
    prob = ODEProblem((u, p, t) -> p * u, complex(reshape(ρ0, :)), extrema(times), L) #ODE 'parameters' are just the liouvillian matrix L
    return solve(prob; saveat=times, kwargs...)
end

function bloch_redfield_ode_dynamics(H, a_ops, ρ0, times; c_ops=[], use_secular=true, secular_cutoff=0.1, kwargs...)
    R, U = bloch_redfield_tensor(H, a_ops; c_ops=c_ops, use_secular=use_secular, secular_cutoff=secular_cutoff)
    ρ0_eb = to_Heb(ρ0, U)
    sol = ode_dynamics(R, ρ0_eb, times; kwargs...)
    N = size(H, 1)
    states = [from_Heb(reshape(sol[:, i], N, N), U) for i in 1:length(sol.t)]
    return states
end

function ode_dynamics_populations(L, ρ0, times; kwargs...)
    sol = ode_dynamics(L, ρ0, times; save_idxs=diagind(ρ0))
    return real(sol[:, :])
end

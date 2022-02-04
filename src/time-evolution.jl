


# ----------------------------------------------- Lindblad Liouvillian ----------------------------------------------- #


# Inspired by QuantumOpticsBase.jl's liouvillian implementation but adapted to work with Zygote auto-diff

spre(op::AbstractMatrix) = kron(I(size(op, 1)), transpose(op))
spost(op::AbstractMatrix) = kron(op, I(size(op, 1)))
dagger(op::AbstractMatrix) = conj(transpose(op))

function liouvillian(H, c_ops)
    L = spre(-1im*H) + spost(1im*H)
    for op in c_ops
        op_dagger = dagger(op)
        op_dagger_op = 0.5*op_dagger*op
        L -= spre(op_dagger_op) + spost(op_dagger_op)
        L += spre(op) * spost(op_dagger)
    end
    return L
end


function bloch_redfield_tensor(H::AbstractMatrix, a_ops::Array; c_ops=[], use_secular=true, secular_cutoff=0.1)
    
    # Use the energy eigenbasis
    H_evals, transf_mat = eigen(Hermitian(complex(H))) # H must be complex so that ChainRules.eigen_rev rule works correctly
    
    #Define function for transforming to Hamiltonian eigenbasis
    to_Heb(op, U) = inv(U) * complex(op) * U

    N = length(H_evals) #Hilbert space dimension
    K = length(a_ops) #Number of system-env interation operators

    # Calculate Liouvillian for Lindblad terms (this also includes unitary dynamics part dρ/dt = -i[H, ρ])
    Heb = to_Heb(H, transf_mat)
    L = liouvillian(Heb, [to_Heb(op, transf_mat) for op in c_ops])
    
    #If only Lindblad collapse terms (no a_ops given) then we're done
    if K==0
        return L #Liouvillian is in the energy eigenbasis here
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
    return L+R

end #Function




function expLt(H, c_ops, t, ρ0)
    L = liouvillian(H, c_ops)
    ρt_vec = exp(L*t)*reshape(ρ0, :)
    ρt = reshape(ρt_vec, size(ρ0)...)
    return ρt
end


#Can't use nullspace method here because nullspace(::ComplexMatrix) internally uses non-julia functions for svd calc (unlike nullspace(::RealMatrix) case)
function steady_state(L, ρ0)
    ρ0_vec = reshape(ρ0, :)
    vals, vecs = eigen(L)
    idxs = findall(abs.(vals) .< 1e-15)
    ss_vec = sum(vecs[:, i] * vecs[:, i]' * ρ0_vec for i in idxs)
    ss = Array(reshape(ss_vec, size(ρ0)))
    return ss /= tr(ss)
end


#Float/Complex errors are sometimes encountered in Zygote if H is real here, but adding complex(H) seems to be a workaround
function bloch_redfield_steady_state(H, a_ops, ρ0; kwargs...)
    U = eigvecs(Hermitian(complex(H))) #Get eigenbasis transformation matrix -- H must be complex so that ChainRules.eigen_rev rule works correctly
    # _, U = eigen(Hermitian(H)) #Get eigenbasis transformation matrix
    R = bloch_redfield_tensor(H, a_ops; kwargs...)
    ρ0_eb = inv(U) * ρ0 * U #Transform to eigenbasis
    ss_eb = steady_state(R, ρ0_eb)
    return U * ss_eb * inv(U) #Transform back from eigenbasis
end


function ode_dynamics(L, ρ0, times; kwargs...)
    prob = ODEProblem((u, p, t) -> p * u, complex(reshape(ρ0, :)), extrema(times), L) #ODE 'parameters' are just the liouvillian matrix L
    return solve(prob; saveat=times, kwargs...)
end 


# ------------------------------------------- Thermal (Gibbs) steady state ------------------------------------------- #

function gibbs_state(H, T)
    A = exp(-e*H/(kb*T))
    B = tr(A)
    return A/B
end

gibbs_dist(H, T) = diag(gibbs_state(H, T))
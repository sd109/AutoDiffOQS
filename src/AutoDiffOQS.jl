
module AutoDiffOQS

using LinearAlgebra, Zygote

export liouvillian, dagger, gibbs_state, gibbs_dist


const kb = 1.380649e-23
const e = 1.602176634e-19


function gibbs_state(H, T)
    A = exp(-e*H/(kb*T))
    B = tr(A)
    return A/B
end

gibbs_dist(H, T) = diag(gibbs_state(H, T))



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



end #module
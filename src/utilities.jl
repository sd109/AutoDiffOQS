
# Physical constants
const kb = 1.380649e-23
const e = 1.602176634e-19

const sigma_x = [0 1; 1 0]
const sigma_y = [0 -im; im 0]
const sigma_z = [1 0; 0 -1]

# projection(N, i, j) = sparse([i], [j], [1.0], N, N) #Sparse arrays don't play nice with Zygote always
# projection(N, i, j) = setindex!(Zygote.Buffer(zeros(N, N)), i, j, 1) |> copy
function projection(N, i, j) 
    #Returns operator |i><j|
    op = Zygote.bufferfrom(zeros(N, N))
    op[i, j] = 1
    return copy(op)
end

to_Heb(op, U) = inv(U) * op * U
from_Heb(op, U) = U * op * inv(U)
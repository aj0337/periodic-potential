module Hamiltonians

# packages #
using SparseArrays
###############################################################################

"""
Creates tight-binding Hamiltonian (as sparse matrix) from neighbors, where
    neighbors[k,i][:] is list with labels of k-th nearest neighbors of site i

< Notes >
-------------------------------------------------------------------------------
- hoppings
hoppings[k] gives hopping between k-th nearest neighbors
-------------------------------------------------------------------------------

GC, 01-06-2022
"""
function TB(neighbors;hoppings=[1.0])
    # Hamiltonian dimension
    dimH = size(neighbors,2)

    # initialize Hamiltonian
    H = spzeros(dimH,dimH)

    # hoppings
    lines = Int64[]
    cols = Int64[]
    vals = Float64[]
    for itk in 1:min(size(neighbors,1),length(hoppings))
        if hoppings[itk]!=0.0
            lineaux = vcat([[it1 for it2 in 1:length(neighbors[itk,it1])] for
                it1 in 1:dimH]...)
            append!(lines,lineaux)

            colaux = vcat(neighbors[itk,:]...)
            append!(cols,colaux)

            tk = hoppings[itk]
            valaux = tk*ones(length(colaux))
            append!(vals,valaux)
        end
    end
    H += sparse(lines,cols,vals,dimH,dimH)

    return H
end

"""
Creates function for tight-binding 1D Bloch Hamiltonian (as sparse matrix) from
    neighbors_uc and neighbors_n, where:

    neighbors_uc[k,i][:] is list with labels of k-th nearest neighbors of site
        i within the unit cell

    neighbors_n[k,i][:] is matrix where first column has labels of k-th nearest
        neighbors of site i from unit cell n, and second column has the
        corresponding n

< Notes>
-------------------------------------------------------------------------------
- usage
H1D(phi) = Hamiltonians.TB_Bloch1D(phi,neighbors_uc,neighbors_n;hoppings)

- phi
phi = k.a
-------------------------------------------------------------------------------

GC, 07-06-2022
"""
function TB_Bloch1D(phi,neighbors_uc,neighbors_n;hoppings=[1.0])
    # Hamiltonian of unit cell
    Huc = TB(neighbors_uc;hoppings=hoppings)

    # Hamiltonian dimension
    dimH = size(Huc,1)

    # Bloch part of Hamiltonian
    HBloch = spzeros(dimH,dimH)
    lines = Int64[]
    cols = Int64[]
    vals = ComplexF64[]
    for itk in 1:min(size(neighbors_n,1),length(hoppings))
        if hoppings[itk]!=0.0
            lineaux = vcat([[it1 for it2 in 1:size(neighbors_n[itk,it1],1)]
                for it1 in 1:size(neighbors_n,2)]...)
            append!(lines,lineaux)

            colaux = vcat(neighbors_n[itk,:]...)
            append!(cols,colaux[:,1])

            tk = hoppings[itk]
            for it in 1:size(colaux,1)
                n = colaux[it,2]
                valaux = tk*exp(im*n*phi)
                push!(vals,valaux)
            end
        end
    end
    HBloch += sparse(lines,cols,vals,dimH,dimH)

    # total Hamiltonian
    H = Huc + HBloch

    return H
end

"""
Creates function for tight-binding 2D Bloch Hamiltonian (as sparse matrix) from
    neighbors_uc and neighbors_n1n2, where:

    neighbors_uc[k,i][:] is list with labels of k-th nearest neighbors of
        site i within the unit cell

    neighbors_n1n2[k,i][:] is matrix where first column has labels of k-th
        nearest neighbors of site i from unit cell (n1,n2), and second and
        third columns have the corresponding n1 and n2, respectively

< Notes>
-------------------------------------------------------------------------------
- usage
H2D(phi1,phi2) = Hamiltonians.TB_Bloch2D(phi1,phi2,neighbors_uc,
    neighbors_n1n2;hoppings)

- phi
phi1 = k.a1
phi2 = k.a2
-------------------------------------------------------------------------------

GC, 13-06-2022
"""
function TB_Bloch2D(phi1,phi2,neighbors_uc,neighbors_n1n2;hoppings=[1.0])
    # Hamiltonian of unit cell
    Huc = TB(neighbors_uc;hoppings=hoppings)

    # Hamiltonian dimension
    dimH = size(Huc,1)

    # Bloch part of Hamiltonian
    HBloch = spzeros(dimH,dimH)
    lines = Int64[]
    cols = Int64[]
    vals = ComplexF64[]
    for itk in 1:min(size(neighbors_n1n2,1),length(hoppings))
        if hoppings[itk]!=0.0
            lineaux = vcat([[it1 for it2 in 1:size(neighbors_n1n2[itk,it1],1)]
                for it1 in 1:size(neighbors_n1n2,2)]...)
            append!(lines,lineaux)

            colaux = vcat(neighbors_n1n2[itk,:]...)
            append!(cols,colaux[:,1])

            tk = hoppings[itk]
            for it in 1:size(colaux,1)
                n1 = colaux[it,2]
                n2 = colaux[it,3]
                valaux = tk*exp(im*n1*phi1)*exp(im*n2*phi2)
                push!(vals,valaux)
            end
        end
    end
    HBloch += sparse(lines,cols,vals,dimH,dimH)

    # total Hamiltonian
    H = Huc + HBloch

    return H
end

###############################################################################
end

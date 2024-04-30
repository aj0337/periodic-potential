module AuxFunctions

# packages #
using LinearAlgebra
using SparseArrays
using Arpack
###############################################################################

"""
For every site of structure supported on 2D honeycomb lattice, gives labels of
    its neighbors up to kmax-th nearest neighbor

< Notes >
-------------------------------------------------------------------------------
- dNN
nearest neighbor distance of honeycomb lattice

- kmax
as of now, must be <= 3

- tol
numerical tolerance for distances

- output
neighbors[k,i][:] gives list with labels of k-th nearest neighbors of site i
-------------------------------------------------------------------------------

GC, 24-08-2023
"""
function neighbors_honeycomb(sites;dNN=1.0,kmax=3,tol=1e-8*dNN)
    # distance of nearest neighbors (ordered)
    # #
    """
        This part could be extended to allow for kmax > 3
    """
    dk = [1,sqrt(3),2]*dNN
    # #

    # neighbors
    neighbors = Array{Vector{Int64}}(undef,kmax,size(sites,1))
    for it1 in 1:size(sites,1)
        list = [[] for itk in 1:kmax]
        for it2 in 1:size(sites,1)
            if it2!=it1
                dist = norm(sites[it1,:]-sites[it2,:])

                if dist < dk[end]+tol
                    for itk in 1:kmax
                        if abs(dist-dk[itk]) < tol
                            push!(list[itk],it2)

                            break
                        end
                    end
                end
            end
        end
        for itk in 1:kmax
            neighbors[itk,it1] = list[itk]
        end
    end

    return neighbors
end

"""
Assumption: unit cell supported on 2D honeycomb lattice and lattice vector
    defining 1D crystal supported on honeycomb lattice.
For every site of unit cell, gives labels of its neighbors (within unit cell or
    for others, keeping track) up to kmax-th nearest neighbor.

< Notes >
-------------------------------------------------------------------------------
- output
neighbors_uc[k,i][:] is list with labels of k-th nearest neighbors of site i
    within the unit cell

neighbors_n[k,i][:] is matrix where first column has labels of k-th nearest
    neighbors of site i from unit cell n, and second column has the
    corresponding n
-------------------------------------------------------------------------------

GC, 07-06-2022
"""
function neighbors_honeycomb_1DBloch(uc,a;dNN=1.0,kmax=3,tol=1e-8*dNN)
    # neighbors within unit cell
    neighbors_uc = neighbors_honeycomb(uc,dNN=dNN,kmax=kmax,tol=tol)

    # distance of nearest neighbors (ordered)
    # #
    """
        This part could be extended to allow for kmax > 3
    """
    dk = [1,sqrt(3),2]*dNN
    # #

    # intercell neighbors
    neighbors_n = Array{Matrix{Int64}}(undef,kmax,size(uc,1))
    for it1 in 1:size(uc,1)
        list1 = [[] for itk in 1:kmax]
        list2 = [[] for itk in 1:kmax]
        for it2 in 1:size(uc,1)
            itn = 1
            while true
                dist = norm(uc[it1,:]-uc[it2,:]-itn*a)

                if dist < dk[end]+tol
                    for itk in 1:kmax
                        if abs(dist-dk[itk]) < tol
                            push!(list1[itk],it2)
                            push!(list2[itk],itn)

                            break
                        end
                    end
                    itn += sign(itn)
                else
                    if itn>0
                        itn = -1
                    else
                        break
                    end
                end
            end
        end
        for itk in 1:kmax
            neighbors_n[itk,it1] = hcat(list1[itk],list2[itk])
        end
    end

    return neighbors_uc,neighbors_n
end

"""
Assumption: unit cell supported on 2D honeycomb lattice and lattice vectors
    defining 2D crystal supported on honeycomb lattice.
For every site of unit cell, gives labels of its neighbors (within unit cell or
    for others, keeping track) up to kmax-th nearest neighbor.

< Notes >
-------------------------------------------------------------------------------
- output
neighbors_n1n2[k,i][:] is matrix where first column has labels of k-th nearest
    neighbors of site i from unit cell (n1,n2), and second and third columns
    have the corresponding n1 and n2, respectively
-------------------------------------------------------------------------------

GC, 15-02-2023
"""
function neighbors_honeycomb_2DBloch(uc,a1,a2;dNN=1.0,kmax=3,tol=1e-8*dNN)
    # neighbors within unit cell
    neighbors_uc = neighbors_honeycomb(uc,dNN=dNN,kmax=kmax,tol=tol)

    # distance of nearest neighbors (ordered)
    # #
    """
        This part could be extended to allow for kmax > 3
    """
    dk = [1,sqrt(3),2]*dNN
    # #
    
    # intercell neighbors
    neighbors_n1n2 = Array{Matrix{Int64}}(undef,kmax,size(uc,1))
    for it1 in 1:size(uc,1)
        list1 = [[] for itk in 1:kmax]
        list2 = [[] for itk in 1:kmax]
        list3 = [[] for itk in 1:kmax]
        for it2 in 1:size(uc,1)
            # #
            """
                This part could be optimized (similarly to the 1D case); the
                    way it is, the code is slower but simpler, and it should
                    always work
            """
            for itn1 in -kmax:kmax, itn2 in -kmax:kmax
                if itn1!=0 || itn2!=0
                    dist = norm(uc[it1,:]-uc[it2,:]-itn1*a1-itn2*a2)

                    if dist < dk[end]+tol
                        for itk in 1:kmax
                            if abs(dist-dk[itk]) < tol
                                push!(list1[itk],it2)
                                push!(list2[itk],itn1)
                                push!(list3[itk],itn2)
                            end
                        end
                    end
                end
            end
            # #
        end
        for itk in 1:kmax
            neighbors_n1n2[itk,it1] = hcat(list1[itk],list2[itk],list3[itk])
        end
    end

    return neighbors_uc,neighbors_n1n2
end

"""
Diagonalization of Hamiltonian (or any other hermitian matrix)

< Notes >
-------------------------------------------------------------------------------
- neigs
nr of eigenvalues to keep (default: size(H,1))
-------------------------------------------------------------------------------

GC, 04-08-2023
"""
function eigensolver_H(H;neigs=size(H,1))
    # check if H is hermitian
    #=
    if ishermitian(H)!=true
        print("ERROR: H is not hermitian")
        return nothing,nothing
    end
    =#
    H_hermitian = (H + H')/2
    if isapprox(H,H_hermitian) != true
        print("ERROR: H is not hermitian")
        return nothing,nothing
    end

    # diagonalization (ordered by smallest eigenvalue)
    En,psin = eigen(Hermitian(Matrix(H)),1:neigs)

    return En,psin
end

"""
Lanczos diagonalization of Hamiltonian (or any other hermitian matrix)

< Notes >
-------------------------------------------------------------------------------
- neigs
nr of eigenvalues to keep (default: 2)
-------------------------------------------------------------------------------

GC, 16-08-2023
"""
function Lanczos_H(H;neigs=2,maxiter=300)
    # check if H is hermitian
    H_hermitian = (H + H')/2
    if isapprox(H,H_hermitian) != true
        print("ERROR: H is not hermitian")
        return nothing,nothing
    end

    # diagonalization (ordered by smallest eigenvalue)
    En,psin = eigs(H_hermitian,nev=neigs,which=:SR,maxiter=maxiter)

    return En,psin
end

"""
phi-path corresponding to k-path -π/a -> 0 -> π/a of 1D lattice

< Notes >
-------------------------------------------------------------------------------
- Nphi
dphi = π/Npi

- output
pathticks = (pathtickpos,pathticklabels), where pathticklabels is list of
    labels of phi points, and pathtickpos the corresponding positions in
    phipath
-------------------------------------------------------------------------------

GC, 07-06-2022
"""
function phipath_1D_mpi0pi(;Npi=100)
    # dphi
    dphi = pi/Npi

    # initialize phi-path
    phipath = Float64[]

    # phi-path points
    mpi = -pi
    z = 0
    ppi = pi

    # -π -> 0
    mpi0 = [phi for phi in mpi:dphi:z-1e-8]
    append!(phipath,mpi0)

    # 0 -> π
    zpi = [phi for phi in z:dphi:ppi-1e-8]
    append!(phipath,zpi)

    # include last point
    append!(phipath,ppi)

    # path ticks
    pathtickpos = [1,length(mpi0)+1,length(phipath)]
    pathticklabels = ["-π","0","π"]
    pathticks = (pathtickpos,pathticklabels)

    return phipath,pathticks
end

"""
Calculates energy bands of H1D(phi) (1D Bloch Hamiltonian) along phipath

< Notes >
-------------------------------------------------------------------------------
- nbands
nr of bands to keep (default: size(H1D(0),1))

- outputs
En_phipath[itphi,:] is list of energies in point phipath[itphi]
-------------------------------------------------------------------------------

GC, 03-11-2022
"""
function En_phipath_1D(H1D,phipath;nbands=size(H1D(0),1))
    # energy bands
    En_phipath = Matrix{Float64}(undef,length(phipath),nbands)
    for itphi in 1:length(phipath)
        phi = phipath[itphi]

        H = H1D(phi)

        En,_ = eigensolver_H(H,neigs=nbands)

        En_phipath[itphi,:] = En
    end

    return En_phipath
end

"""
phi-path corresponding to k-path M -> K -> Γ -> K' -> M of 2D hexagonal lattice
    assuming a1 = [1/2,sqrt(3)/2]*a, a2 = [-1/2,sqrt(3)/2]*a

< Notes >
-------------------------------------------------------------------------------
- Nphi
dphi = (π/3)/Npiover3

- output
phipath[:,1(2)] is list of phi1 (phi2) points
-------------------------------------------------------------------------------

GC, 13-06-2022
"""
function phipath_2D_MKGammaKpM(;Npiover3=10)
    # dphi
    dphi = (pi/3)/Npiover3

    # initialize phi-path
    phi1path = Float64[]
    phi2path = Float64[]

    # phi-path points
    M = [pi, -pi]
    K = [2*pi/3, -2*pi/3]
    Gamma = [0.0, 0.0]
    Kp = [-2*pi/3, 2*pi/3]
    M2 = [-pi, pi] #equivalent to M

    #=
        M -> K -> Γ -> K' -> M can be parametrized as
            (phi1,phi2) = (phi,-phi), with phi going from pi to -pi
    =#

    #M -> K
    MK_phi1 = [phi1 for phi1 in M[1]:-dphi:K[1]+1e-8]
    MK_phi2 = [phi2 for phi2 in M[2]:dphi:K[2]-1e-8]
    append!(phi1path,MK_phi1)
    append!(phi2path,MK_phi2)

    #K -> Gamma
    KGamma_phi1 = [phi1 for phi1 in K[1]:-dphi:Gamma[1]+1e-8]
    KGamma_phi2 = [phi2 for phi2 in K[2]:dphi:Gamma[2]-1e-8]
    append!(phi1path,KGamma_phi1)
    append!(phi2path,KGamma_phi2)

    #Gamma -> K'
    GammaKp_phi1 = [phi1 for phi1 in Gamma[1]:-dphi:Kp[1]+1e-8]
    GammaKp_phi2 = [phi2 for phi2 in Gamma[2]:dphi:Kp[2]-1e-8]
    append!(phi1path,GammaKp_phi1)
    append!(phi2path,GammaKp_phi2)

    #K' -> M2
    KpM2_phi1 = [phi1 for phi1 in Kp[1]:-dphi:M2[1]+1e-8]
    KpM2_phi2 = [phi2 for phi2 in Kp[2]:dphi:M2[2]-1e-8]
    append!(phi1path,KpM2_phi1)
    append!(phi2path,KpM2_phi2)

    # include last point
    append!(phi1path,M2[1])
    append!(phi2path,M2[2])

    # phi-path
    phipath = hcat(phi1path,phi2path)

    # path ticks
    pathtickpos = [1,size(MK_phi1,1)+1,1+size(MK_phi1,1)+size(KGamma_phi1,1),
        1+size(MK_phi1,1)+size(KGamma_phi1,1)+size(GammaKp_phi1,1),
        size(phi1path,1)]
    pathticklabels = ["M","K","Γ","K'","M"]
    pathticks = (pathtickpos,pathticklabels)

    return phipath,pathticks
end

"""
phi-path corresponding to k-path Γ -> K -> M -> Γ of 2D hexagonal lattice
    assuming a1 = [1/2,sqrt(3)/2]*a, a2 = [-1/2,sqrt(3)/2]*a

GC, 08-05-2023
"""
function phipath_2D_GammaKMGamma(;Npiover3=10)
    # dphi
    dphi = (pi/3)/Npiover3

    # initialize phi-path
    phi1path = Float64[]
    phi2path = Float64[]

    # phi-path points
    Gamma = [0.0, 0.0]
    K = [2*pi/3, -2*pi/3]
    M = [pi, -pi]
    M2 = [-pi, 0.0] #equivalent to M

    #Gamma -> K
    GammaK_phi1 = [phi1 for phi1 in Gamma[1]:dphi:K[1]-1e-8]
    GammaK_phi2 = [phi2 for phi2 in Gamma[2]:-dphi:K[2]+1e-8]
    append!(phi1path,GammaK_phi1)
    append!(phi2path,GammaK_phi2)

    #K -> M
    KM_phi1 = [phi1 for phi1 in K[1]:dphi:M[1]-1e-8]
    KM_phi2 = [phi2 for phi2 in K[2]:-dphi:M[2]+1e-8]
    append!(phi1path,KM_phi1)
    append!(phi2path,KM_phi2)

    #M2 -> Gamma
    npts = round(sqrt(3)/2*length(GammaK_phi1))
    dphiaux = (Gamma[1]-M[2])/npts
    M2Gamma_phi1 = [phi1 for phi1 in M2[1]:dphiaux:Gamma[1]-sign(dphiaux)*1e-8]
    M2Gamma_phi2 = [M2[2] for phi1 in M2[1]:dphiaux:Gamma[1]-sign(dphiaux)*1e-8]
    append!(phi1path,M2Gamma_phi1)
    append!(phi2path,M2Gamma_phi2)

    # include last point
    append!(phi1path,Gamma[1])
    append!(phi2path,Gamma[2])

    # phi-path
    phipath = hcat(phi1path,phi2path)
    
    # path ticks
    pathtickpos = [1,
        1+size(GammaK_phi1,1),
        1+size(GammaK_phi1,1)+size(KM_phi1,1),
        size(phi1path,1)]
    pathticklabels = ["Γ","K","M","Γ"]
    pathticks = (pathtickpos,pathticklabels)

    return phipath,pathticks
end

"""
Calculates energy bands of H2D(phi1,phi2) (2D Bloch Hamiltonian) along phipath

GC, 03-11-2022
"""
function En_phipath_2D(H2D,phipath;nbands=size(H2D(0,0),1))
    # energy bands
    En_phipath = Matrix{Float64}(undef,size(phipath,1),nbands)
    for itphi in 1:size(phipath,1)
        phi1 = phipath[itphi,1]
        phi2 = phipath[itphi,2]

        H = H2D(phi1,phi2)

        En,_ = eigensolver_H(H,neigs=nbands)

        En_phipath[itphi,:] = En
    end

    return En_phipath
end

"""
Diagonalizes H2D(phi1,phi2) (2D Bloch Hamiltonian) along Brillouin zone, using
    Monkhorst-Pack grid

< Notes >
-------------------------------------------------------------------------------
- NBZ1 (NBZ2)
nr of phi1 (phi2) points
-------------------------------------------------------------------------------

GC, 15-02-2023
"""
function En_psin_phi_BZ_2D(H2D;NBZ1=5,NBZ2=5,nbands=size(H2D(0,0),1))
    # phiBZ (Monkhorst-Pack grid)
    dphi1 = 2*pi/NBZ1
    phi1BZ = [phi for phi in 0:dphi1:2*pi-1e-8]
    dphi2 = 2*pi/NBZ2
    phi2BZ = [phi for phi in 0:dphi2:2*pi-1e-8]
    NBZ = NBZ1*NBZ2

    # diagonalization
    En_BZ = Matrix{Float64}(undef,NBZ,nbands)
    psin_BZ = Array{ComplexF64}(undef,NBZ,size(H2D(0,0),1),nbands)
    phi_BZ = Matrix{Float64}(undef,NBZ,2)
    for itphi1 in 1:NBZ1, itphi2 in 1:NBZ2
        itphi = (itphi1-1)*NBZ2 + itphi2

        phi1 = phi1BZ[itphi1]
        phi2 = phi2BZ[itphi2]

        H = H2D(phi1,phi2)

        En,psin = eigensolver_H(H,neigs=nbands)

        En_BZ[itphi,:] = En
        psin_BZ[itphi,:,:] = psin
        phi_BZ[itphi,:] = [phi1,phi2]
    end

    return En_BZ,psin_BZ,phi_BZ
end

###############################################################################
end

module MyPyPlots

# packages #
using PyPlot
###############################################################################

"""
Scatter plot of 2D structures

< Notes >
-------------------------------------------------------------------------------
- neighbors
to plot lines between neighbors, take neighbors=neighbors where
    neighbors[k,i][:] is list with labels of k-th nearest neighbors of site i

- linecolors
useful tip: linecolors=["blue",(0,0,0,0),"red"] will only show 1st and 3rd
    neighbor lines (in blue and red, respectively)

- vectors
to plot 2D vectors v1, v2, ..., take vecs = [[v1x,v1y], [v2x,v2y], ...]

- site labels
if sitelabels=true, shows site labels

- save figure
if save=true, saves plot as <fig_dir_name>.svg (default: 
    /Users/scgo/Downloads/p.svg)
-------------------------------------------------------------------------------

GC, 26-09-2023
"""
function structures2D(sites;markersize="automatic",
    neighbors=nothing,linecolors="automatic",linewidth="automatic",
    vecs=[],
    sitelabels=false,
    save=false,fig_dir_name="/Users/scgo/Downloads/p")

    # markersize
    if markersize=="automatic"
        xmin = minimum(sites[:,1])
        xmax = maximum(sites[:,1])
        ymin = minimum(sites[:,2])
        ymax = maximum(sites[:,2])
        
        dx = xmax-xmin
        dy = ymax-ymin
        
        markersize = 1000/maximum([dx,dy])^2
    end
    
    # main plot
    scatter(sites[:,1], sites[:,2],
        s=markersize,
        c="gray")
    axis("equal")
    axis("off")

    # linewidth
    if linewidth=="automatic"
        xmin = minimum(sites[:,1])
        xmax = maximum(sites[:,1])
        ymin = minimum(sites[:,2])
        ymax = maximum(sites[:,2])
        
        dx = xmax-xmin
        dy = ymax-ymin
        
        linewidth = 4/maximum([dx,dy])
    end
    
    # neighbors
    if typeof(neighbors)==Matrix{Vector{Int64}}
        if linecolors=="automatic"
            # #
            """
                add more colors if needed
            """
            colors = ["blue","red","green","purple","orange"]
            # #
            sizecolors = typemax(Int)
        else
            colors = linecolors
            sizecolors = length(colors)
        end

        # #
        """
            This part is slow and can be optimized
        """
        for itk in 1:min(size(neighbors,1),sizecolors)
            for it1 in 1:size(sites,1)
                for it2 in neighbors[itk,it1]
                    x = [sites[it1,1],sites[it2,1]]
                    y = [sites[it1,2],sites[it2,2]]

                    plot(x, y, color=colors[itk], linewidth=linewidth)
                end
            end
        end
        # #
    end

    # vectors
    for vec in vecs
        arrow(0.0,0.0,vec[1],vec[2],
            lw=linewidth,head_width=0.3,head_length=0.3,
            color="black")
    end
    
    # site labels
    if sitelabels==true
        for i in 1:size(sites,1)
            annotate(i,(sites[i,1],sites[i,2]))
        end
    end
    
    # save figure
    if save==true
        savefig(fig_dir_name*".svg")
    end
    
    # show figure
    show()

    return 
end

"""
Plot of energy levels

< Notes >
-------------------------------------------------------------------------------
- EF
shifts zero energy to EF (Fermi energy)
-------------------------------------------------------------------------------

GC, 26-09-2023
"""
function En(En;markersize=4,
    EF=0,
    xlims="auto",ylims="auto",
    imgsize=(6,6),
    save=false,fig_dir_name="/Users/scgo/Downloads/p")
    
    #EF
    if EF!=0
        En -= EF*ones(length(En))
    end
    
    # main plot
    figure(figsize=imgsize)
    plot(1:length(En),En,markersize=markersize,
        linestyle="None",marker="o",color="black")
    xlabel("Principal quantum number")
    ylabel("Energy")
    if xlims != "auto"
        xlim(xmin=xlims[1],xmax=xlims[2])
    end
    if ylims != "auto"
        ylim(ymin=ylims[1],ymax=ylims[2])
    end
    
    # save figure
    if save==true
        savefig(fig_dir_name*".svg")
    end
    
    # show figure
    show()

    return 
end

"""
Scatter plot of wavefunction at each site of 2D structure, |ψ(i)|

< Notes >
-------------------------------------------------------------------------------
- phase
if phase=true (useful for complex wavefunctions or to distinguish signs), plots
    complex phases with arrows
-------------------------------------------------------------------------------

GC, 25-07-2023
"""
function wave2D(sites,psi;markersize="automatic",
    neighbors=nothing,linecolors="automatic",linewidth="automatic",
    phase=false,arrowlength=0.5,
    sitelabels=false,
    save=false,fig_dir_name="/Users/scgo/Downloads/p")

    # markersize
    if markersize=="automatic"
        xmin = minimum(sites[:,1])
        xmax = maximum(sites[:,1])
        ymin = minimum(sites[:,2])
        ymax = maximum(sites[:,2])
        
        dx = xmax-xmin
        dy = ymax-ymin
        
        markersize = 10000/maximum([dx,dy])^2
    end
    
    # main plot
    psi2 = real(psi.*conj(psi))
    abs_psi = psi2.^(1/2)
    scatter(sites[:,1], sites[:,2],
        s=markersize*abs_psi,
        c="black")
    axis("equal")
    axis("off")
    
    # linewidth
    if linewidth=="automatic"
        xmin = minimum(sites[:,1])
        xmax = maximum(sites[:,1])
        ymin = minimum(sites[:,2])
        ymax = maximum(sites[:,2])
        
        dx = xmax-xmin
        dy = ymax-ymin
        
        linewidth = 4/maximum([dx,dy])
    end
    
    # neighbors
    if typeof(neighbors)==Matrix{Vector{Int64}}
        if linecolors=="automatic"
            # #
            """
                add more colors if needed
            """
            colors = ["blue","red","green","purple","orange"]
            # #
            sizecolors = typemax(Int)
        else
            colors = linecolors
            sizecolors = length(colors)
        end

        # #
        """
            This part is slow and can be optimized
        """
        for itk in 1:min(size(neighbors,1),sizecolors)
            for it1 in 1:size(sites,1)
                for it2 in neighbors[itk,it1]
                    x = [sites[it1,1],sites[it2,1]]
                    y = [sites[it1,2],sites[it2,2]]

                    plot(x, y, color=colors[itk], linewidth=linewidth)
                end
            end
        end
        # #
    end

    # phase
    if phase==true
        angles = [angle(psi[i]) for i in 1:length(psi)]
        for i in 1:length(psi)
            if abs_psi[i]>1e-8
                x = sites[i,1]
                y = sites[i,2]
                dx = arrowlength*cos(angles[i])
                dy = arrowlength*sin(angles[i])
                arrow(x,y,dx,dy,
                    lw=linewidth,head_width=0.15,head_length=0.15,
                    color="black")
            end
        end
    end
    
    # site labels
    if sitelabels==true
        for i in 1:size(sites,1)
            annotate(i,(sites[i,1],sites[i,2]))
        end
    end
    
    # save figure
    if save==true
        savefig(fig_dir_name*".svg")
    end
    
    # show figure
    show()

    return 
end

"""
Plot of energy bands

< Notes >
-------------------------------------------------------------------------------
- pathticks
to have xticks, take pathticks = (pathtickpos,pathticklabels), where
    pathticklabels is list of labels of phi points, and pathtickpos the
    corresponding positions in phipath
-------------------------------------------------------------------------------

GC, 26-07-2023
"""
function Ebands(En_phipath;linewidth=2,
    linecolor="auto",
    ylims="auto",
    imgsize=(6,6),
    pathticks=([],[]),
    EF=0,
    save=false,fig_dir_name="/Users/scgo/Downloads/p")

    # EF
    if EF!=0
        En_phipath -= EF*ones(size(En_phipath))
    end

    # main plot
    figure(figsize=imgsize)
    if linecolor=="auto"
        for it in 1:size(En_phipath,2)
            plot(1:size(En_phipath,1), En_phipath[:,it], linewidth=linewidth)
        end
    else
        for it in 1:size(En_phipath,2)
            plot(1:size(En_phipath,1), En_phipath[:,it], linewidth=linewidth, 
                    c=linecolor)
        end
    end
    xlabel("k-path")
    ylabel("Energy")
    if ylims != "auto"
        ylim(ymin=ylims[1],ymax=ylims[2])
    end
    xticks(pathticks[1],pathticks[2])

    # save figure
    if save==true
        savefig(fig_dir_name*".svg")
    end
    
    # show figure
    show()

    return
end

"""
Plot of Sz(i) in 2D structure, using red/blue color for positive/negative

GC, 26-07-2023
"""
function Szi_2D(sites,Szi;markersize="automatic",
    neighbors=nothing,linecolors="automatic",linewidth="automatic",
    sitelabels=false,
    save=false,fig_dir_name="/Users/scgo/Downloads/p")

    # linewidth
    if linewidth=="automatic"
        xmin = minimum(sites[:,1])
        xmax = maximum(sites[:,1])
        ymin = minimum(sites[:,2])
        ymax = maximum(sites[:,2])
        
        dx = xmax-xmin
        dy = ymax-ymin
        
        linewidth = 4/maximum([dx,dy])
    end
    
    # neighbors
    if typeof(neighbors)==Matrix{Vector{Int64}}
        if linecolors=="automatic"
            # #
            """
                add more colors if needed
            """
            colors = ["blue","red","green","purple","orange"]
            # #
            sizecolors = typemax(Int)
        else
            colors = linecolors
            sizecolors = length(colors)
        end

        # #
        """
            This part is slow and can be optimized
        """
        for itk in 1:min(size(neighbors,1),sizecolors)
            for it1 in 1:size(sites,1)
                for it2 in neighbors[itk,it1]
                    x = [sites[it1,1],sites[it2,1]]
                    y = [sites[it1,2],sites[it2,2]]

                    plot(x, y, color=colors[itk], linewidth=linewidth)
                end
            end
        end
        # #
    end

    # markersize
    if markersize=="automatic"
        xmin = minimum(sites[:,1])
        xmax = maximum(sites[:,1])
        ymin = minimum(sites[:,2])
        ymax = maximum(sites[:,2])
        
        dx = xmax-xmin
        dy = ymax-ymin
        
        markersize = 10000/maximum([dx,dy])^2
    end
    
    # main plot
    absSzi = [abs(Szi[i]) for i in 1:length(Szi)]
    markercolor = [Szi[i]>0 ? "red" : "blue" for i in 1:length(Szi)]
    scatter(sites[:,1], sites[:,2],
        s=markersize*absSzi,
        c=markercolor)
    axis("equal")
    axis("off")

    # site labels
    if sitelabels==true
        for i in 1:size(sites,1)
            annotate(i,(sites[i,1],sites[i,2]))
        end
    end
    
    # save figure
    if save==true
        savefig(fig_dir_name*".svg")
    end
    
    # show figure
    show()

    return
end

"""
Plot of energy levels for spin-up/dn

GC, 26-07-2023
"""
function En_updn(En_up,En_dn;markersize=4,
    xlims="auto",ylims="auto",
    imgsize=(6,6),
    EF=0,
    legends=true,
    save=false,fig_dir_name="/Users/scgo/Downloads/p")
    
    # EF
    if EF!=0
        En_up -= EF*ones(size(En_up))
        En_dn -= EF*ones(size(En_dn))
    end
    
    # main plot
    figure(figsize=imgsize)
    plot(1:length(En_up),En_up,markersize=markersize,
        linestyle="None",marker="o",color="red",
        label="up")
    plot(1:length(En_dn),En_dn,markersize=markersize,
        linestyle="None",marker="x",color="blue",
        label="dn")
    xlabel("Principal quantum number")
    ylabel("Energy")
    if xlims != "auto"
        xlim(xmin=xlims[1],xmax=xlims[2])
    end
    if ylims != "auto"
        ylim(ymin=ylims[1],ymax=ylims[2])
    end
    if legends==true
        legend()
    end
    
    # save figure
    if save==true
        savefig(fig_dir_name*".svg")
    end
    
    # show figure
    show()

    return 
end

"""
Plot of energy bands for spin-up/dn

< Notes >
-------------------------------------------------------------------------------
- dashed_dn
if dashed_dn=true (useful for spin degenerate bands), dn bands are dashed
-------------------------------------------------------------------------------

GC, 26-07-2023
"""
function Ebands_updn(En_up_phipath,En_dn_phipath;linewidth=2,
    ylims="auto",
    imgsize=(6,6),
    pathticks=([],[]),
    EF=0,
    dashed_dn=false,
    legends=true,
    save=false,fig_dir_name="/Users/scgo/Downloads/p")

    # EF
    if EF!=0
        En_up_phipath -= EF*ones(size(En_up_phipath))
        En_dn_phipath -= EF*ones(size(En_dn_phipath))
    end

    # dashed_dn
    if dashed_dn==true
        linestyle="dashed"
    else
        linestyle="solid"
    end
    
    # main plot
    figure(figsize=imgsize)
    plot(1:size(En_up_phipath,1), En_up_phipath[:,1], 
            linewidth=linewidth, c="red", label="up")
    for it in 2:size(En_up_phipath,2)
        plot(1:size(En_up_phipath,1), En_up_phipath[:,it], 
            linewidth=linewidth, c="red")
    end
    plot(1:size(En_dn_phipath,1), En_dn_phipath[:,1], 
            linewidth=linewidth, c="blue", label="dn",
            linestyle=linestyle)
    for it in 2:size(En_dn_phipath,2)
        plot(1:size(En_dn_phipath,1), En_dn_phipath[:,it], 
            linewidth=linewidth, c="blue", linestyle=linestyle)
    end
    xlabel("k-path")
    ylabel("Energy")
    if ylims != "auto"
        ylim(ymin=ylims[1],ymax=ylims[2])
    end
    xticks(pathticks[1],pathticks[2])
    if legends==true
        legend()
    end

    # save figure
    if save==true
        savefig(fig_dir_name*".svg")
    end
    
    # show figure
    show()
    
    return
end

"""
Plot of LDOS(x,y)

GC, 27-06-2023
"""
function LDOS_xy(xrange,yrange,LDOS;
    colorscale=true,
    save=false,fig_dir_name="/Users/scgo/Downloads/p")

    # main plot
    contourf(xrange,yrange,LDOS,cmap="inferno",levels=100,extend="both")
    axis("equal")
    axis("off")
    if colorscale==true
        colorbar()
    end
    
    # save figure
    if save==true
        savefig(fig_dir_name*".png")
    end
    
    # show figure
    show()

    return
end

"""
Scatter plot of 3D structures

< Notes >
-------------------------------------------------------------------------------
- elev,azim,roll
parameters to control perspective
-------------------------------------------------------------------------------

GC, 24-08-2023
"""
function structures3D(sites;markersize="automatic",
    imgsize=(6,6),elev=15,azim=0,roll=0,
    neighbors=nothing,linecolors="automatic",linewidth="automatic",
    vecs=[],
    sitelabels=false,
    save=false,fig_dir_name="/Users/scgo/Downloads/p")

    # markersize
    if markersize=="automatic"
        xmin = minimum(sites[:,1])
        xmax = maximum(sites[:,1])
        ymin = minimum(sites[:,2])
        ymax = maximum(sites[:,2])
        zmin = minimum(sites[:,3])
        zmax = maximum(sites[:,3])
        
        dx = xmax-xmin
        dy = ymax-ymin
        dz = zmax-zmin
        
        markersize = 1000/maximum([dx,dy,dz])^2
    end
    
    # main plot
    fig = plt.figure(figsize=imgsize)
    ax = fig.add_subplot(projection="3d")
    ax.scatter(sites[:,1], sites[:,2], sites[:,3],
        s=markersize,
        c="gray")
    ax.set_box_aspect([maximum(sites[:,1])-minimum(sites[:,1]), 
        maximum(sites[:,2])-minimum(sites[:,2]), 
        maximum(sites[:,3])-minimum(sites[:,3])])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=elev,azim=azim,roll=roll)

    # linewidth
    if linewidth=="automatic"
        xmin = minimum(sites[:,1])
        xmax = maximum(sites[:,1])
        ymin = minimum(sites[:,2])
        ymax = maximum(sites[:,2])
        zmin = minimum(sites[:,3])
        zmax = maximum(sites[:,3])
        
        dx = xmax-xmin
        dy = ymax-ymin
        dz = zmax-zmin
        
        linewidth = 4/maximum([dx,dy,dz])
    end
    
    # neighbors
    if typeof(neighbors)==Matrix{Vector{Int64}}
        if linecolors=="automatic"
            # #
            """
                add more colors if needed
            """
            colors = ["blue","red","green","purple","orange"]
            # #
            sizecolors = typemax(Int)
        else
            colors = linecolors
            sizecolors = length(colors)
        end

        # #
        """
            This part is slow and can be optimized
        """
        for itk in 1:min(size(neighbors,1),sizecolors)
            for it1 in 1:size(sites,1)
                for it2 in neighbors[itk,it1]
                    x = [sites[it1,1],sites[it2,1]]
                    y = [sites[it1,2],sites[it2,2]]
                    z = [sites[it1,3],sites[it2,3]]

                    ax.plot(x, y, z, color=colors[itk], linewidth=linewidth)
                end
            end
        end
        # #
    end

    # vectors
    for vec in vecs
        ax.quiver(0.0,0.0,0.0,vec[1],vec[2],vec[3],
            color="black")
    end
    
    # site labels
    if sitelabels==true
        for i in 1:size(sites,1)
            ax.text(sites[i,1],sites[i,2],sites[i,3],i)
        end
    end
    
    # save figure
    if save==true
        savefig(fig_dir_name*".svg")
    end
    
    # show figure
    show()

    return 
end

###############################################################################
end
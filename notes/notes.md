## Insights

1. ARPACK (used for computing eigenvalues of a sparse matrix in wanniertools. Note that this is also what scipy uses when you call eigsh) will lead to spurious eigenvalues and therefore spurious jumps in the bands. This seems to be a known issue as discussed in "[arpack-issues](https://scicomp.stackexchange.com/questions/36767/accuracy-issues-with-arpack-in-julia-for-eigenvalues-of-smallest-magnitude)"

   ![alt text](notes/image.png)

The issue seems to stem from the fact that arpack is buggy when looking for eigenvalues around 0. So it is best to set the energy around which the eigenvalues need to be computed to a small non-zero value. In wannier tools, one can control this using the E_Fermi value in wt.in. Since wt.in sets it to 0 by default, remember to change it to 0.0001 for instance. In numpy, the way to fix the issue is to set the sigma parameter in eigsh to a similar small non-zero value. This issue will only reduce the spikes, not eliminate it completely.

2. When using scipy sparse solver, there is a k parameter which is the number of eigenvalues to be computed. However, in sparse mode. I find that it is not enough to set k to the number of bands one is interested in. If you are interested in the first 8 bands around Fermi and set k = 8, a large portion of the bands will likely be missing. I find setting k to a large number like k = 120 allows complete bands and less spurious points. Also note that k can have a maximum value of number of atoms in the system - 2.

3. The effect of adding an onsite term is that it shifts the Dirac cone to that point in energy. For e.g. if the onsite energy is set to -0.25 eV, the Dirac cone will now be at -0.25 eV. This needs to be considered when setting the E_fermi parameter in wt.in. It needs to be set to -0.25 eV if one is interested in the bands around the Dirac cone.

# Main
In the code directory, the following relevant code scripts that are available, are from the Nature paper "Geometric constraints on human brain function" by Pang et al.: 
- surface_eigenmodes.py: Python script to calculate the geometric eigenmodes of a cortical surface.
- demo_eigenmode_analysis.sh: shell script to demonstrate how to calculate geometric eigenmodes. This script calls surface_eigenmodes.py.
- demo_eigenmode_analysis.m: MATLAB script to demonstrate how to use eigenmodes to analyze fMRI data.
- demo_connectome_eigenmode_calculation.m: MATLAB script to demonstrate how to calculate connectome, connectome density-matched, and EDR eigenmodes.
- generate_paper_figures_main_Nature.m: MATLAB script to generate the main figures of the Nature paper.

In the results directory, the recon_beta, reconstructed data, and eigenmodes of the geometric, EDR, and hybrid model are found.

The other code scripts are: 
- hybrid_model.m: MATLAB script to develop a hybrid model combining the connectivity with the geometric characteristics.
- variational_free_energy.m: MATLAB script with derivation of the variational free energy principle used to compute the Bayesian model comparison.

 

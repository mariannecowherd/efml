# Phase-resolved wave boundary layer dynamics in a shallow estuary

Data analysis and visualizations associated with preprint: https://doi.org/10.1002/essoar.10506048.1
 
The 'data' folder contains .npy files with saved outputs from data processing steps. For full unprocessed data, see data repository hosted at https://web.stanford.edu/group/efml-data-repo/


`decomp.py` performs wave phase decomposition of the timeseries Vectrino data
`fit_vel_ens_gm.py` takes phase-decomposed ensembles and fits the profiles to the Grant-Madsen model (Grant & Madsen 1979) to produce Figure 1
`ustar_omega.py` examines scaling between boundary layer thickness from the observations to friction velocity normalized by anngular wave frequency to produce Figure 2
`delta_nut_measured_modeled` compares observations of eddy viscosity with other eddy viscosity estimates from scaling and from the General Ocean Turbulence Model to produce Figure 3

`vectrinofuncs.py` contains standalone functions used in other steps, including velocity rotation, Hilbert transform-based phase decomposition, spectral decomposition, dissipation estimation, and interpolation

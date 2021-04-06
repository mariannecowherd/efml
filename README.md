# Phase-resolved wave boundary layer dynamics in a shallow estuary

This repo contains analysis and visualizations performed for the paper **[Phase-resolved wave boundary layer dynamics in a shallow estuary](https://doi.org/10.1029/2020GL092251)** in Geophysical Research Letters by Cowherd et al., 2021.

## Data
The 'data' folder contains .npy files with saved outputs from data processing steps. For full unprocessed data, see data repository hosted at https://purl.stanford.edu/wv787xr0534

The GOTM NetCDF output required to make Figure 3 can be downloaded here:
https://drive.google.com/drive/folders/1ubC_8dzXPfYH_62-ECe9mgpx5CQXjfQF?usp=sharing

## Analysis

`decomp.py` performs wave phase decomposition of the timeseries Vectrino data

`fit_vel_ens_gm.py` takes phase-decomposed ensembles and fits the profiles to the Grant-Madsen model (Grant & Madsen 1979) to produce Figure 1

`ustar_omega.py` examines scaling between boundary layer thickness from the observations to friction velocity normalized by anngular wave frequency to produce Figure 2

`delta_nut_measured_modeled_nondim_final` compares observations of eddy viscosity with other eddy viscosity estimates from scaling and from the General Ocean Turbulence Model to produce Figure 3. This requires the GOTM data output (see above).

`vectrinofuncs.py` contains standalone functions used in other steps, including velocity rotation, Hilbert transform-based phase decomposition, spectral decomposition, dissipation estimation, and interpolation


## Guide
0. Clone the repository using git clone https://github.com/mariannecowherd/efml.git
### From raw data
1. Download the VectrinoSummer folder from the data repository and unzip
2. Edit all entries `dirs.py` to reflect your local file structure
3. Run `decomp.py` to conduct the phase-decomposition
4. Run other scripts (see "Analysis") to produce figures in any order

### From pre-processed data
1. Edit `dir_home` and `dir_plots` in `dirs.py` to reflect your local file structure
2. Run scripts (see "Analysis") to produce figures in any order. (Not `decomp.py` as this requires the raw data).

## License
Data and scripts here are free to use for noncommercial purposes. 

Please cite the following paper for the analysis steps and modeling data:
Cowherd, M., Egan, G., Monismith, S., & Fringer, O. (2021). [Phase‐resolved wave boundary layer dynamics in a shallow estuary.](https://doi.org/10.1029/2020GL092251) Geophysical Research Letters.

Please cite the data repository and/or related papers as relevant for field data:

[Data repository](https://purl.stanford.edu/wv787xr0534)

Egan, G., Cowherd, M., Fringer, O., & Monismith, S. (2019). [Observations of near‐bed shear stress in a shallow, wave‐and current‐driven flow.](https://doi.org/10.1029/2019JC015165) Journal of Geophysical Research: Oceans.

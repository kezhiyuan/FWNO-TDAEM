# Data sources

## Musgrave, South Australia

The 3D resistivity model corresponds to the isotropic inversion described by Su et al. (2021), *3D Electromagnetic Inversion of Anisotropic Resistivity Using a Finite-Difference Method*, *Minerals*, 11(2), 218. The package contains `iso.rho`, its cropped counterpart `iso-new.rho`, and the 396 FWNO/FD inputs derived from those files. Cite the original study when using the model. See the original paper for the field-data access statement and contact for the original survey files. [Su et al. (2021)](https://doi.org/10.3390/min11020218)

The paired FD and FWNO files contain calculated positive response amplitudes at 50 m in nT/s. Geological polygons are attributed to the Geological Survey of South Australia under CC BY 4.0. The locator geometry is from Natural Earth. Source details are retained under `03_results/musgrave/geology`.

## Synthetic deterministic inversion examples

Both examples use 400 positions, one flight height of 30 m, and 16 time gates. The 20 × 20 layouts are subsampled from a denser survey and have 60 m and 120 m adjacent spacings. The model arrays use a 124 × 124 × 20 grid and the active region is `[23:101, 23:101, :]` in zero-based Python indexing. `model_raw` is log10 conductivity. Convert it to resistivity in Ω·m with `10**(-model_raw)`.

The frozen FWNO generated the synthetic clean responses from the truth models. The observation files include the clean and noisy responses, error standard deviations, and survey geometry. The recovery folders retain the final arrays and SSIM scores. The checkerboard recovery is a continuation from the earlier 400-sounding run. The ring recovery starts from a uniform model.

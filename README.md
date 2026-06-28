# neural-graphs-mice-development

This repository is a brach of the GitLab Repository from Rodrigo C. G. Pena: https://gitlab.com/ceda-unibas/neural-graphs-mice-development

Comparing the structure of neural connections during the development of mice.

[![license][license-badge]][license]

## Project status

Ongoing

## Installation

Create a new conda environment for this project by running the following on your Terminal:

```sh
conda env create -f environment.yml
```

Activate the new environment with `conda activate ngmd` (or `source activate ngmd`, depending on your system) and install the necessary dependencies with

```sh
make install
```

Alternatively, you may explicitly run `pip3 install .`

## Configuration

Project paths and analysis parameters are configured in `configs/config.yaml`.
The default file has this structure:

```yaml
paths:
  data_dir: "F:\\Experiments\\Steffen\\Intrinsic_Imaging"
  models_dir: ""

precompute_sttc_arrays:
  seed: 2024
  dt: 0.25
  method: "pulse_shuffle"
  num_trials: 3
  n_jobs: 1
  force_recompute: false

network_summaries:
  num_subsample_trials: 20
  subsample_size: 250

structural_consistency:
  num_trials: 20
  num_subsample_trials: 20
  subsample_size: 250
```

Set `paths.data_dir` to the folder containing the `DON-*` dataset folders.
Set `paths.models_dir` if you use notebooks or analyses that load trained
models. The environment variables `NGMD_DATA` and `NGMD_MODELS` override those
two path values at runtime.

Use `subsample_size: null` in YAML to run network summaries or structural
consistency on full graphs instead of node-subsampled graphs.

## Usage

The scripts and notebooks load YAML configuration through `ngmd.config`. The
combined notebook `notebooks/06_all_analysis_summary.ipynb` can run the
analysis scripts for missing outputs and generate multi-panel summary plots in
`<data_dir>/analysis_files/summary_plots`.

## Contributing

See [CONTRIBUTING][contributing].

## Authors and acknowledgment

This project is a result of a collaboration between [CeDA][ceda] and [Flavio Donato's research group][flavio-donato] at the University of Basel.

## License

This project is released under the [BSD 3-Clause License][license].

[ceda]: https://ceda.unibas.ch/
[contributing]: CONTRIBUTING.md
[flavio-donato]: https://www.biozentrum.unibas.ch/facilities/services/services-a-z/overview/unit/research-group-flavio-donato
[license]: LICENSE
[license-badge]: https://img.shields.io/badge/license-BSD-green

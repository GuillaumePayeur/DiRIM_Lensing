# DiRIM_Lensing

[![Paper: Coming Soon](https://img.shields.io/badge/paper-coming%20soon-lightgrey)](#paper-and-links)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

DiRIM_Lensing models strong gravitational lenses by combining denoising diffusion and recurrent inference machines in a single framework that allows sampling of the source and convergence map jointly as pixelated images for a given lens observation.

**Authors:** Guillaume Payeur, Laurence Perreault-Levasseur, Gabriel Missael Barco, Yashar Hezaveh

![DiRIM lens reconstructions](results/figures_github/summary.gif)

## Why This Repository Exists

This repository is published primarily for research transparency. It provides the implementation used to run the experiments and produce results for the associated article.

## Paper and Links

- **Paper title:** _Strong Gravitational Lensing Posterior Sampling in Pixel-Space Using Diffusion Models and Recurrent Inference Machines_
- **Preprint (arXiv):** _[TBA]_
- **Journal / conference version:** _[TBA]_

## Installation

Install the package:

```bash
pip install -e .
```

Install package + script dependencies (recommended if you run anything in `scripts/`):

```bash
pip install -e ".[scripts]"
```

Install everything (scripts + dev tooling):

```bash
pip install -e ".[all]"
```

## Repository Structure

- `src/dirim_lensing`: source code.
- `scripts`: dataset generation, training, posterior sampling, and analysis scripts.
- `configs`: YAML experiment configurations used by scripts.

## Citation

If you use this code in your research, please cite the corresponding article.

```bibtex
@article{payeur2026dirim_lensing,
	title   = {Strong Gravitational Lensing Posterior Sampling in Pixel-Space Using Diffusion Models and Recurrent Inference Machines},
	author  = {Payeur, Guillaume and Perreault-Levasseur, Laurence and Barco, Gabriel Missael and Hezaveh, Yashar},
	year    = {2026}
}
```

## License

This project is released under the MIT License.

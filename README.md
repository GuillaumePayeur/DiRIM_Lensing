# DiRIM_Lensing

[![Status: Research Code](https://img.shields.io/badge/status-research%20code-blue)](#)
[![Paper: Coming Soon](https://img.shields.io/badge/paper-coming%20soon-lightgrey)](#paper-and-links)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

DiRIM_Lensing models strong gravitational lenses by combining denoising diffusion and recurrent inference machines in a single framework that allows sampling of the source and convergence map jointly as pixelated images for a given lens observation.

**Authors:** Guillaume Payeur, Laurence Perreault-Levasseur, Yashar Hezaveh

![DiRIM lens reconstructions](results/figures_github/summary.png)

## Why This Repository Exists

This repository is published primarily for research transparency. It provides the implementation used to run the experiments and produce results for the associated article.

## Paper and Links

- **Paper title:** _[TBA]_
- **Preprint (arXiv):** _[TBA]_
- **Journal / conference version:** _[TBA]_

## Installation

```bash
pip install -e .
```

## Repository Structure

- `src/dirim_lensing`: core package implementation.
- `scripts`: training, testing, and analysis scripts.
- `configs`: experiment configurations.

## Citation

If you use this code in your research, please cite the corresponding article.

```bibtex
% TODO: Replace with final BibTeX entry
@article{payeur2026dirim_lensing,
	title   = {Title to be added},
	author  = {Payeur, Guillaume and Perreault-Levasseur, Laurence and Hezaveh, Yashar},
	year    = {2026}
}
```

## License

This project is released under the MIT License.

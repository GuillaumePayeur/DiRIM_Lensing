# DiRIM_Lensing

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

DiRIM for lensing: Posterior sampling of source galaxy and foreground mass distribution as pixelated images in strong gravitational lensing using diffusion-based generative models and recurrent inference machines.

**Authors:** Guillaume Payeur, Laurence Perreault-Levasseur, Gabriel Missael Barco, Yashar Hezaveh

![DiRIM lens model](summary.gif)

## Why This Repository Exists

This repository is published for transparency. It provides the source code and scripts used to train and test our models.

## Paper and Links

- **Title:** _Strong Gravitational Lensing Posterior Sampling in Pixel-Space Using Diffusion Models and Recurrent Inference Machines_
- **Preprint (arXiv):** _[TBA]_
- **Journal / conference version:** _[TBA]_

## Installation

If you are setting this up from scratch, run the following before `pip install -e`:

```bash
git clone https://github.com/GuillaumePayeur/DiRIM_Lensing
cd DiRIM_Lensing
conda create -n dirim-lensing python=3.12 -y
conda activate dirim-lensing
```

Install the package:

```bash
pip install -e .
```

Install package + script dependencies (recommended if you run anything in `scripts/`):

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
@unpublished{payeur2026gravitational,
	title   = {Strong Gravitational Lensing Posterior Sampling in Pixel-Space Using Diffusion Models and Recurrent Inference Machines},
	author  = {Payeur, Guillaume and Perreault-Levasseur, Laurence and Barco, Gabriel Missael and Hezaveh, Yashar},
	year    = {2026}
}
```

## License

This project is released under the MIT License.

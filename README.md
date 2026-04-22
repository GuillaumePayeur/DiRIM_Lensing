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

```bash
# Clone and enter the repository
git clone https://github.com/GuillaumePayeur/DiRIM_Lensing
cd DiRIM_Lensing

# Create a python environment
conda create -n dirim-lensing python=3.12 -y
conda activate dirim-lensing
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia # GPU installation of PyTorch for Windows or Lunix

# Install the package
pip install -e . # package only
pip install -e ".[all]" # package + script dependencies (recommended if you run anything in `scripts/`)
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

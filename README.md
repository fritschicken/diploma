# Proximity-induced magnetism on topological edge states

Code and documents for my Physics MSc thesis at Eötvös Loránd University (ELTE), 2026.

**Author:** László Erik Fritsch
**Supervisors:** László Oroszlány, Zoltán Tajkov (Department of Physics of Complex Systems, ELTE)

The thesis studies a 2D topological insulator, described by the Bernevig–Hughes–Zhang (BHZ) model, coupled to a magnetic layer. It looks at how the topological edge states change the magnetic exchange interactions in that layer.

## Repository contents

| Path | Description |
| --- | --- |
| [`diploma.pdf`](diploma.pdf) | The thesis |
| [`ppt.pdf`](ppt.pdf) | Slides from the thesis defence |
| [`lib/`](lib/) | `magnetic_models`, the Python package with the tight-binding model used in the thesis |

## Installation

Install the package straight from GitHub:

```bash
pip install "git+https://github.com/fritschicken/diploma.git#subdirectory=lib"
```

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/fritschicken/diploma.git
cd diploma
pip install -e "lib[analysis]"
```

The `analysis` extra installs the packages used by the thesis calculations: `grogupy` for exchange interactions, `z2pack` for Chern numbers, `mpi4py`, `pandas`, `seaborn`, `scikit-learn` and others.

See [`lib/README.md`](lib/README.md) for how to use the package.

## License

The code is released under the [MIT License](lib/LICENSE).

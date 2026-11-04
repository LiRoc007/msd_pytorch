[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3560114.svg)](https://doi.org/10.5281/zenodo.3560114)

# Mixed-scale Dense Networks for PyTorch

An implementation of Mixed-Scale Dense networks in PyTorch.

* Free software: GNU General Public License v3
* Documentation: [https://ahendriksen.github.io/msd_pytorch]

## Getting Started

It takes a few steps to setup Mixed-scale Dense Networks for PyTorch
on your machine. We recommend installing [Anaconda package
manager](https://www.anaconda.com/download/) for Python 3.

### Requirements

This port targets **PyTorch 2.15.0** and Python 3.11 or newer.
It requires an NVIDIA GPU, a CUDA-enabled PyTorch installation, the matching
CUDA toolkit (including `nvcc`), and a C++20-compatible compiler. Linux x64 and
Windows x64 are supported. On Windows, install Visual Studio C++ Build Tools
and run the build from an **x64 Native Tools Command Prompt**.

The original memory-efficient CUDA kernels, symmetric boundary padding, model
architecture, and state-dict keys are preserved. The custom kernels support
`float32` and `float64`; CPU execution and automatic mixed precision are not
supported.

### Installing with Conda

The old prebuilt Conda packages target PyTorch 1.8.1. For this port, create an
environment with `conda create -n msd-torch215 python=3.12 pip`, activate it,
and use the source installation below.
The legacy `conda/` recipes and GitLab release jobs are for old releases;
this port uses the pip source build.

### Installing from source

To install msd_pytorch from source, you need to have the CUDA toolkit
installed. Specifically, you need `nvcc` and a compatible C++
compiler. Moreover, you need to have a working installation of
PyTorch.

Clone this repository:

```bash
git clone https://github.com/LiRoc007/msd_pytorch.git
cd msd_pytorch
```

Install PyTorch first. For CUDA 13.4, for example:

```bash
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu134
python -m pip install setuptools wheel ninja
python -m pip install --no-build-isolation -e .
```

Use the [PyTorch installer](https://pytorch.org/get-started/locally/) for other
CUDA builds. The toolkit used to compile this extension should match
`torch.version.cuda`. Set `CUDA_HOME` if it is not detected, and `CXX` if a
particular host compiler is required. Build isolation is disabled above so the
extension is built against the PyTorch already installed in your environment.
On machines without a visible GPU during compilation, set
`TORCH_CUDA_ARCH_LIST` to the deployment GPU's compute capability.

Run the tests on a CUDA GPU:

```bash
python -m pip install pytest
python -m pytest msd_pytorch/tests -m "not slow"
python -m pytest msd_pytorch/tests/test_torch_compatibility.py
```

This includes 2D/3D forward and backward comparisons with native PyTorch
autograd, frozen parameters, non-default CUDA streams, and the existing
regression/segmentation training and checkpoint tests.

### Running the examples

To learn more about the functionality of the package check out our
examples folder.


## Cite
If you find our work useful, please cite as:

```
@software{hendriksen-2019-msd-pytor,
  author       = {Hendriksen, Allard A.},
  title        = {ahendriksenh/msd\_pytorch: v0.7.2},
  month        = dec,
  year         = 2019,
  publisher    = {Zenodo},
  version      = {v0.7.2},
  doi          = {10.5281/zenodo.3560114},
  url          = {https://doi.org/10.5281/zenodo.3560114}
}
```

## Authors and contributors

* **Allard Hendriksen** - *Initial work*
* **Ryan Pollitt** - *Port CUDA convolution code from 2D to 3D!*
* **Jonas Adler** - *Discussions and code*
* **Richard Schoonhoven** - *Testing and patches*

See also the list of [contributors](https://github.com/ahendriksen/msd_pytorch/contributors) who participated in this project.

## How to contribute

Contributions are always welcome. Please submit pull requests against the `dev` branch.

If you have any issues, questions, or remarks, then please open an issue on GitHub.

## License

This project is licensed under the GNU General Public License v3 - see the [LICENSE.md](LICENSE.md) file for details.

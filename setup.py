#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os.path
import os
import sys

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.md', encoding='utf-8') as history_file:
    history = history_file.read()

with open(os.path.join('msd_pytorch', 'VERSION')) as version_file:
    version = version_file.read().strip()

requirements = [
    "torch>=2.15",
    "numpy",
    "tifffile",
    "imageio"
]

dev_requirements = [
    'autopep8',
    'rope',
    'jedi',
    'flake8',
    'importmagic',
    'autopep8',
    'black',
    'yapf',
    'snakeviz',
    # Documentation
    'sphinx',
    'sphinx_rtd_theme',
    'recommonmark',
    # Other
    'watchdog',
    'coverage',
    'pytest',
]

# Kernels are instantiated in separate CUDA translation units.
nvcc_args = ['--static-global-template-stub=false']
if sys.platform == 'win32':
    nvcc_args += ['-Xcompiler', '/Zc:preprocessor,/utf-8']


setup(
    author="Allard Hendriksen",
    author_email='allard.hendriksen@cwi.nl',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
    ],
    description="An implementation of Mixed-Scale Dense networks in PyTorch. ",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='msd_pytorch',
    name='msd_pytorch',
    packages=find_packages(include=['msd_pytorch']),
    python_requires='>=3.11',
    extras_require={'dev': dev_requirements},
    url='https://github.com/ahendriksen/msd_pytorch',
    version=version,
    zip_safe=False,
    ext_modules=[
        CUDAExtension(
            name='msd_custom_convolutions',
            sources=[
                'msd_pytorch/msd_custom_convolutions.cpp',
                'msd_pytorch/msd_custom_convolutions/torch_cuda_dispatch.cu',
                # 2D kernels:
                'msd_pytorch/msd_custom_convolutions/conv2d_backward_k.cu',
                'msd_pytorch/msd_custom_convolutions/conv2d_backward_bias.cu',
                'msd_pytorch/msd_custom_convolutions/conv2d_backward_x.cu',
                'msd_pytorch/msd_custom_convolutions/conv2d_forward.cu',
                # 3D kernels:
                'msd_pytorch/msd_custom_convolutions/conv3d_backward_bias.cu',
                'msd_pytorch/msd_custom_convolutions/conv3d_backward_k.cu',
                'msd_pytorch/msd_custom_convolutions/conv3d_backward_x.cu',
                'msd_pytorch/msd_custom_convolutions/conv3d_forward.cu',
            ],
            extra_compile_args={
                'cxx': ['/utf-8'] if sys.platform == 'win32' else [],
                'nvcc': nvcc_args,
            },
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
)

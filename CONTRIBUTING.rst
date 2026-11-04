.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/ahendriksen/msd_pytorch/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.


Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/ahendriksen/msd_pytorch/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Building from source
--------------------

Install CUDA-enabled PyTorch 2.15.0, the matching CUDA toolkit, and a
compatible C++ compiler, then run::

    python -m pip install setuptools wheel ninja
    python -m pip install --no-build-isolation -e .[dev]
    python -m pytest msd_pytorch/tests -m "not slow"

On Windows use an x64 Native Tools Command Prompt. On Linux set ``CXX`` if
the default compiler is incompatible with the CUDA toolkit. See README.md
for installation and GPU architecture selection.

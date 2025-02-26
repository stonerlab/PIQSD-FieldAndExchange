# Sources for: Path integral spin dynamics with exchange and external field

This repository contains source code which implements a path integral approach to calculate the thermodynamics of two quantum spins coupled by exchange interaction in a magnetic field. It contains a numerical approach based on atomistic spin dynamics, and in the special case where the magnetic field is along the z-direction, exact diagonalisation results are provided for the quantum system. 

## Authors

Thomas Nussle, *School of Physics and Astronomy, University of Leeds, Leeds, LS2 9JT, United Kingdom*, http://orcid.org/0000-0001-9783-4287

Stam Nicolis, *Institut Denis Poisson, Université de Tours, Université d'Orléans, CNRS (UMR7013), Parc de Grandmont, F-37200, Tours, France*, http://orcid.org/0000-0002-4142-5043

Iason Sofos, *School of Physics and Astronomy, University of Leeds, Leeds, LS2 9JT, United Kingdom*, http://orcid.org/0009-0006-3666-0416

Joseph Barker, *School of Physics and Astronomy, University of Leeds, Leeds, LS2 9JT, United Kingdom*, http://orcid.org/0000-0003-4843-5516

## Description

This is a simple code for atomistic simulations for two spins coupled by exchange interaction in a magnetic field. It computes both the classical limit and quantum corrections of the thermal expectation value of the z-component of the total spin. 

The approximation scheme is a high temperature approach, either explicitly by doing a Taylor series in the inverse temperature for the computed effective Hamiltonian or hidden in always required exponential series of the quantum partition function. The effective Hamiltonian is computed either directly from this series or as a difference to the classical limit of the system, which is slightly more accurate in the case of ferromagnetically coupled spins.

In the special case where the magnetic field is along the z-direction, exact diagonalisation results are used to compute thermal expectation values for the whole temperature range.

Exact results from exact diagonalisation serve as a reference where they can be computed.


## File descriptions

### ./

**environment.yml**
Conda environment file to reproduce the python environment for executing the calculations.

**LICENSE**
MIT License file.

**README.md**
This readme file.

### ./python/

This folder contains python code and scripts to generate the figures.

**python/asd.py** 
Defines python functions for atomistic spin dynamics calculations including numerical integration methods, effective fields and stochastic fields. 

**python/effective_field_computation.py** 
Defines python functions for symbolical calculations leading to the effective field (numerical) used in the atomistic spin dynamics calculations.

**python/exact_diagonalisation.py** 
Defines python functions for obtaining the diagonalised Hamiltonian with eigenvalues and eigenvectors used in the atomistic spin dynamics code, when appropriate, or as exact results for comparison.

**python/figure1.py** 
Calculates and plots the quantum analytic result for the thermal expectation value of two spins s=1/2 with ferromagnetic exchange J=1 in units of g$\mu_B$, using results from exact diagonalisation. Expectation value of Sz for s=1/2 as a function of temperature.

**python/figure3.py** 
Calculates and plots the quantum analytic results and approximate results for the classical limit, first and second quantum corrections using the logarithmic expression of the field, after taking the difference from the expected classical limit for said field using the atomistic approximation method. Expectation value of Sz for s=2 as a function of temperature.

**python/figure4{a,b}.py** 
Calculates and plots the quantum analytic results and exact quantum field of the atomistic approximation method using results from exact diagonalisation. Expectation value of Sz for s={1/2,2} as a function of temperature.

**python/figure5{a,b}.py** 
Calculates and plots the quantum analytic results and exact quantum field of the atomistic approximation method using results from exact diagonalisation, this time for antiferromagnetically coupled spins. Expectation value of Sz for s={1/2,2} as a function of temperature.

**python/figure6.py** 
Calculates and plots the quantum analytic results and approximate results for the classical limit, first and second quantum corrections using the high temperature approximation of the atomistic approximation method. Expectation value of Sz for s=2 as a function of temperature.

**python/generating_moments.py** 
Defines python function to compute the moments required to compute the effective field up to a given order from using sympy and non-commutative algebra. Moments up to order 5 coded hard for computational efficiency, but given enough time and computational resources, all moments up to this order and a few higher orders can be computed.

**python/pisd.py**
An executable python program for running general path integral spin dynamics calculations using the functions in python/asd.py.

### ./figures/

Output generated from the python/figure*X*.py scripts. 

**figures/figure*X*.pdf**
PDF figure used in the manuscript.

**figures/figure*X*.log**
Output logged from executing the figure script.

**figures/figure*X*__data_**
Folder containing the raw data generated by the script with the filenames representing `<method>_<approximation>_<spin>.tsv`

### ./resources/

**aps-paper.mplstyle**
matplotlib style file for plotting figures with fonts and font sizes similar to the American Physical Society typesetting. 

## Computational Environment

All calculations were performed on a Mac Studio (Apple M2 Ultra, 64GB RAM, Model Identifier: Mac14,14) running macOS version 15.2 (Sequoia). Python and associated packages were installed using conda. The installed package versions were:
 - python=3.10.16
 - matplotlib=3.6.2
 - numba=0.56.4
 - numpy=1.23.5
 - scipy=1.9.3
 - sympy=1.11.1
  
The conda environment can be recreated using the `environment.yml` file on .

<details>
  <summary>Click here for the complete list of package installed by conda including all dependencies and hashes</summary>
```text
# Name                    Version                   Build  Channel
brotli                    1.1.0                hd74edd7_2    conda-forge
brotli-bin                1.1.0                hd74edd7_2    conda-forge
bzip2                     1.0.8                h99b78c6_7    conda-forge
ca-certificates           2024.12.14           hf0a4a13_0    conda-forge
certifi                   2024.12.14         pyhd8ed1ab_0    conda-forge
contourpy                 1.3.1           py310h7f4e7e6_0    conda-forge
cycler                    0.12.1             pyhd8ed1ab_1    conda-forge
fonttools                 4.55.3          py310hc74094e_1    conda-forge
freetype                  2.12.1               hadb7bae_2    conda-forge
gmp                       6.3.0                h7bae524_2    conda-forge
gmpy2                     2.1.5           py310h805dbd7_3    conda-forge
kiwisolver                1.4.7           py310h7306fd8_0    conda-forge
lcms2                     2.16                 ha0e7c42_0    conda-forge
lerc                      4.0.0                h9a09cb3_0    conda-forge
libblas                   3.9.0           20_osxarm64_openblas    conda-forge
libbrotlicommon           1.1.0                hd74edd7_2    conda-forge
libbrotlidec              1.1.0                hd74edd7_2    conda-forge
libbrotlienc              1.1.0                hd74edd7_2    conda-forge
libcblas                  3.9.0           20_osxarm64_openblas    conda-forge
libcxx                    19.1.7               ha82da77_0    conda-forge
libdeflate                1.23                 hec38601_0    conda-forge
libffi                    3.4.2                h3422bc3_5    conda-forge
libgfortran               5.0.0           13_2_0_hd922786_3    conda-forge
libgfortran5              13.2.0               hf226fd6_3    conda-forge
libjpeg-turbo             3.0.0                hb547adb_1    conda-forge
liblapack                 3.9.0           20_osxarm64_openblas    conda-forge
libllvm11                 11.1.0               hfa12f05_5    conda-forge
liblzma                   5.6.3                h39f12f2_1    conda-forge
libopenblas               0.3.25          openmp_h6c19121_0    conda-forge
libpng                    1.6.45               h3783ad8_0    conda-forge
libsqlite                 3.48.0               h3f77e49_0    conda-forge
libtiff                   4.7.0                h551f018_3    conda-forge
libwebp-base              1.5.0                h2471fea_0    conda-forge
libxcb                    1.17.0               hdb1d25a_0    conda-forge
libzlib                   1.3.1                h8359307_2    conda-forge
llvm-openmp               19.1.7               hdb05f8b_0    conda-forge
llvmlite                  0.39.1          py310h1e34944_1    conda-forge
matplotlib                3.6.2           py310hb6292c7_0    conda-forge
matplotlib-base           3.6.2           py310h78c5c2f_0    conda-forge
mpc                       1.3.1                h8f1351a_1    conda-forge
mpfr                      4.2.1                hb693164_3    conda-forge
mpmath                    1.3.0              pyhd8ed1ab_1    conda-forge
munkres                   1.1.4              pyh9f0ad1d_0    conda-forge
ncurses                   6.5                  h5e97a16_2    conda-forge
numba                     0.56.4          py310h3124f1e_1    conda-forge
numpy                     1.23.5          py310h5d7c261_0    conda-forge
openjpeg                  2.5.3                h8a3d83b_0    conda-forge
openssl                   3.4.0                h81ee809_1    conda-forge
packaging                 24.2               pyhd8ed1ab_2    conda-forge
pillow                    11.1.0          py310h61efb56_0    conda-forge
pip                       24.3.1             pyh8b19718_2    conda-forge
pthread-stubs             0.4               hd74edd7_1002    conda-forge
pyparsing                 3.2.1              pyhd8ed1ab_0    conda-forge
python                    3.10.16         h870587a_1_cpython    conda-forge
python-dateutil           2.9.0.post0        pyhff2d567_1    conda-forge
python_abi                3.10                    5_cp310    conda-forge
readline                  8.2                  h92ec313_1    conda-forge
scipy                     1.9.3           py310ha0d8a01_2    conda-forge
setuptools                75.8.0             pyhff2d567_0    conda-forge
six                       1.17.0             pyhd8ed1ab_0    conda-forge
sympy                     1.11.1          pypyh9d50eac_103    conda-forge
tk                        8.6.13               h5083fa2_1    conda-forge
tornado                   6.4.2           py310h078409c_0    conda-forge
tzdata                    2025a                h78e105d_0    conda-forge
unicodedata2              16.0.0          py310h078409c_0    conda-forge
wheel                     0.45.1             pyhd8ed1ab_1    conda-forge
xorg-libxau               1.0.12               h5505292_0    conda-forge
xorg-libxdmcp             1.1.5                hd74edd7_0    conda-forge
zstd                      1.5.6                hb46c0d2_0    conda-forge
```
</details>

## Reproduction

 The `make` build tool can be used to execute the Makefile and re-produce all of the figures. The steps to reproduce are:

```bash
conda env create -f environment.yml
conda activate quantum_spin_dynamics
make clean
make
```

Note that the the atomistic spin dynamics are stochastic so the results will differ slightly due to random seeding.

### Runtimes

- python/figure1.py: 0.453 (s)
- python/figure3.py: 11219.177 (s)
- python/figure4a.py: 6848.936 (s)
- python/figure4b.py: 76669.533 (s)
- python/figure5a.py: 66635.302 (s)
- python/figure5b.py: 84818.633 (s)
- python/figure6.py: 4259.530 (s)

## Code use for general calculations

The python/pisd.py script can be used to generate data with different integration methods, approximations and spin values.

Using the provided environment.yml file, create a conda environment in your terminal:

```bash
conda env create -f environment.yml
conda activate quantum_spin_dynamics
```

Then run the python/pisd.py script

```bash
python python/pisd.py <options>
```

where the options available are

```text
usage: pisd.py [-h] [--integrator {symplectic}] --approximation {low-temperature,classical-limit,high-temperature-first-order,high-temperature-second-order} --spin SPIN

Simulation parameters from command line.

options:
  -h, --help            show this help message and exit
  --integrator {runge-kutta-4,symplectic}
                        Numerical integration method for solving the spin dynamics
  --approximation {classical-limit, quantum-approximation-sympy, quantum-exact, exact-bz-two-spins}
                        Approximation scheme to use
  --order ORDER         Order of the approximation scheme (up to order 4 for, "quantum-exact" and "quantum-approximation-sympy" methods)
  
  --spin SPIN           Quantum spin value (should normally be an integer multiple of 1/2)
  
  --from_difference True/False  Computing effective Hamiltonian from difference to its classical limit
  
  --exchange EXCHANGE   Value for the exchange parameter J (in units of g mu_B)
```
  
### Additional variables

Depending on computational resources and specific system, one can change some parameters 
in the python/pisd.py script (or examples scripts as well):
- the value of the gilbert damping `alpha` (default is 0.5)
- the value of the applied field by changing `applied_field`. Only use field along z-direction in case of "exact-bz-two-spins"
- the range and increments in temperature for atomistic simulations by 
changing `np.linspace(<starting-temperature>, <final-temperature>, 
<number-of-temperatures-in-range>)`
- the initial orientation for both spins orientation `s0` (must be of unit norm and different 
from `np.array([[0, 0, 1], [0, 0, 1])`
- the equilibration period `equilibration_time`, the computation time `production_time`
and the integration time step `time_step` for each stochastic realisation
- the number of stochastic realisations `num_realisation` of the noise over
which to average the time average

## Notes

The numba package is used for just in time compilation which greatly reduces the calculation time. In principle the `@njit` statements can be removed from all code if numba is not supported on a given platform, but the calculation time will be extremely long.

This work is an extension of the method for a single spin in a constant magnetic field from: Thomas Nussle, Stam Nicolis and Joseph Barker, "Numerical simulations of a spin dynamics model based on a path integral approach", [Phys. Rev. Research 5, 043075 (2023)](https://doi.org/10.1103/PhysRevResearch.5.043075).

This work is an extension of the method for a single spin in a constant magnetic field with a quadratic term for a uniaxial anistropy from: Thomas Nussle, Pascal Thibaudeau and Stam Nicolis, "Path Integral Spin Dynamics for Quantum Paramagnets", [Adv. Phys. Research, 202400057 (2024)](https://doi.org/10.1002/apxr.202400057).

By default, the atomistic spin dynamics uses a symplectic integrator described in: Pascal Thibaudeau and David Beaujouan, "Thermostatting the atomic spin dynamics from controlled demons", [Phys. A: Stat. Mech. its Appl. 391, 1963–1971 (2012)](http://dx.doi.org/10.1016/j.physa.2011.11.030).

## Grant Acknowledgement

This software was produced with funding from the UKRI Engineering and Physical Sciences Research Council [grant number EP/V037935/1 - *Path Integral Quantum Spin Dynamics*] and support from the Royal Society through a University Research Fellowship.
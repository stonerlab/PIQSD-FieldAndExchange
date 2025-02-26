import os
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
import time

# local imports
import exact_diagonalisation


def main():
    plt.style.use('resources/aps-paper.mplstyle')

    j_exchange = 1.0

    b_z = 1.0  # Field z-component in Tesla

    # Temperature parameters
    temperatures = np.linspace(0.01, 10, 100)

    quantum_solution = exact_diagonalisation.exact_thermal_spin_half_s_z(j_exchange, b_z, temperatures)

    # --- plotting ---
    plt.plot(temperatures, quantum_solution, color='#FF0000')

    plt.xlabel(r"$T$ (K)")
    plt.ylabel(r"$\langle\hat{S}_z\rangle$ ($\hbar$)")
    plt.legend(title=r"$s=\frac{1}{2}$")

    plt.savefig('figures/figure1.pdf', transparent=True)
    # plt.show()


if __name__ == "__main__":
    start = time.process_time()
    main()
    end = time.process_time()
    print(f'runtime: {end - start:.3f} (s)')
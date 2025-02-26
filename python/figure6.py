import os
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
import time

# local imports
import asd
import exact_diagonalisation


def main():
    plt.style.use('resources/aps-paper.mplstyle')
    data_path = 'figures/figure6_data'
    os.makedirs(data_path, exist_ok=True)

    quantum_spin = 2.0
    j_exchange_in_g_mu_b = 100.0
    alpha = 0.5  # Gilbert Damping parameter.

    b_z = 1.0  # Field z-component in Tesla
    applied_field = np.array((0, 0, b_z))

    # Temperature parameters
    temperatures_asd = np.linspace(0.1, 1000, 200)
    num_realisation = 5

    # Initial conditions
    s0 = np.array([
        [1 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3)]
        ,
        [1 / np.sqrt(3), 1.0 / np.sqrt(3), -1.0 / np.sqrt(3)]
    ])  # Initial spin

    # Equilibration time, final time and time step
    equilibration_time = 5  # Equilibration time ns
    production_time = 15  # Final time ns
    time_step = 0.000005  # Time step ns, "linspace" so needs to turn num into int

    asd_data_file_1 = f'{data_path}/qsd_symplectic_classical-limit_{quantum_spin}_J={j_exchange_in_g_mu_b}_hamiltonian_classical.txt'
    if os.path.exists(asd_data_file_1):
        temperatures_asd, h_classical_asd = np.loadtxt(asd_data_file_1, unpack=True)
    else:
        solver = asd.solver_factory('symplectic', 'classical-limit', 1, quantum_spin,
                                    applied_field, alpha, time_step, j_exchange_in_g_mu_b)
        h_classical_asd = asd.compute_temperature_dependence_hamiltonian_classical(solver, temperatures_asd, time_step,
                                                    equilibration_time, production_time, num_realisation, s0, applied_field, quantum_spin, j_exchange_in_g_mu_b)

    np.savetxt(f"{data_path}/qsd_symplectic_classical-limit_{quantum_spin}_J={j_exchange_in_g_mu_b}_hamiltonian_classical.txt",
               np.column_stack((temperatures_asd, h_classical_asd)), fmt='%.8e',
               header='temperature_kelvin sz-expectation_hbar')

    # --- plotting ---
    colors = np.array(('#D728A1', '#FF9900', '#DFD620'))
    plt.plot(temperatures_asd, 1/(asd.kB*temperatures_asd)*abs(3*100*asd.g_factor*asd.muB/4+h_classical_asd), label=r"estimate of $\beta\left|\left|\hat{\cal H}-{\cal H}_{classical}\right|\right|_{\infty}$", linestyle=(0, (4, 6)), color=colors[0])

    plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)

    plt.xlabel(r"$T$ (K)")
    plt.ylabel(r"$\beta\left|\left|\hat{\cal H}-{\cal H}_{classical}\right|\right|_{\infty}$ (unitless)")
    plt.legend(title=rf'$s={str(Fraction(quantum_spin))}$')
    plt.axhline(1, color='grey', linestyle='--', linewidth=0.5)
    plt.ylim(0, 10)

    plt.savefig('figures/figure6.pdf', transparent=True)
    # plt.show()


if __name__ == "__main__":
    start = time.process_time()
    main()
    end = time.process_time()
    print(f'runtime: {end - start:.3f} (s)')
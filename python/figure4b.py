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
    data_path = 'figures/figure4b_data'
    os.makedirs(data_path, exist_ok=True)

    quantum_spin = 2
    j_exchange_in_g_mu_b = 1.0

    alpha = 0.5  # Gilbert Damping parameter.

    b_z = 1.0  # Field z-component in Tesla
    applied_field = np.array((0, 0, b_z))

    # Temperature parameters
    temperatures = np.linspace(0.1, 10, 100)
    num_realisation = 5

    # Initial conditions
    s0 = np.array([
        [1 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3)]
        ,
        [-1 / np.sqrt(3), -1.0 / np.sqrt(3), 1.0 / np.sqrt(3)]
    ])  # Initial spin

    # Equilibration time, final time and time step
    equilibration_time = 5  # Equilibration time ns
    production_time = 15  # Final time ns
    time_step = 0.000004  # Time step ns, "linspace" so needs to turn num into int

    # --- calculate solutions and save data ---
    exact_diagonalisation_data_file = f'{data_path}/analytic_quantum_solution_{quantum_spin}_J={j_exchange_in_g_mu_b}.txt'
    if os.path.exists(exact_diagonalisation_data_file):
        temperatures, quantum_solution = np.loadtxt(exact_diagonalisation_data_file, unpack=True)
    else:
        magnetisation = exact_diagonalisation.find_magnetisation(int(quantum_spin * 2), 2)
        quantum_solution = magnetisation(j_exchange_in_g_mu_b, 1.0, temperatures)

    np.savetxt(f"{data_path}/analytic_quantum_solution_{quantum_spin}_J={j_exchange_in_g_mu_b}.txt",
               np.column_stack((temperatures, quantum_solution)), fmt='%.8e',
               header='temperature_kelvin sz-expectation_hbar')

    asd_data_file_classical = f'{data_path}/qsd_symplectic_classical-limit_{quantum_spin}_J={j_exchange_in_g_mu_b}.txt'
    if os.path.exists(asd_data_file_classical):
        temperatures, sz_asd_classical = np.loadtxt(asd_data_file_classical, unpack=True)
    else:
        solver = asd.solver_factory('symplectic', 'classical-limit', 1, quantum_spin,
                                    applied_field, alpha, time_step, j_exchange_in_g_mu_b)
        sz_asd_classical = asd.compute_temperature_dependence(solver, temperatures, time_step,
                                                    equilibration_time, production_time, num_realisation, s0)

    np.savetxt(f"{data_path}/qsd_symplectic_classical-limit_{quantum_spin}_J={j_exchange_in_g_mu_b}.txt",
               np.column_stack((temperatures, sz_asd_classical)), fmt='%.8e',
               header='temperature_kelvin sz-expectation_hbar')

    asd_data_file_order_2 = f'{data_path}/qsd_symplectic_quantum-exact-two-spins-bz_{quantum_spin}_J={j_exchange_in_g_mu_b}.txt'
    if os.path.exists(asd_data_file_order_2):
        temperatures, sz_asd_order_2 = np.loadtxt(asd_data_file_order_2, unpack=True)
    else:
        solver = asd.solver_factory('symplectic', 'exact-bz-two-spins', 1, quantum_spin,
                                    applied_field, alpha, time_step, j_exchange_in_g_mu_b)
        sz_asd_order_2 = asd.compute_temperature_dependence(solver, temperatures, time_step,
                                                    equilibration_time, production_time, num_realisation, s0)

    np.savetxt(f"{data_path}/qsd_symplectic_quantum-exact-two-spins-bz_{quantum_spin}_J={j_exchange_in_g_mu_b}.txt",
               np.column_stack((temperatures, sz_asd_order_2)), fmt='%.8e',
               header='temperature_kelvin sz-expectation_hbar')


    # --- plotting ---
    colors = np.array(('#D728A1', '#FF9900', '#DFD620'))
    plt.plot(temperatures, sz_asd_classical*quantum_spin, label='classical limit', marker=".", linestyle='None', color=colors[0])
    plt.plot(temperatures, sz_asd_order_2*(quantum_spin+1), label='quantum exact field', marker=".", linestyle='None', color=colors[1])
    plt.plot(temperatures, quantum_solution, label='quantum solution', color='#FF0000')

    plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)

    plt.xlabel(r"$T$ (K)")
    plt.ylabel(r"$\langle\hat{S}_z\rangle$ ($\hbar$)")
    plt.legend(title=rf'$s={str(Fraction(quantum_spin))}$')
    plt.text(2, 1.8, '(b)')
    plt.savefig('figures/figure4b.pdf', transparent=True)
    # plt.show()


if __name__ == "__main__":
    start = time.process_time()
    main()
    end = time.process_time()
    print(f'runtime: {end - start:.3f} (s)')
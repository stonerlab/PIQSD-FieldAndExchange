import time
import argparse
import numpy as np
import asd

# Parsing parameters from command line
parser = argparse.ArgumentParser(description='Simulation parameters from command line.')

parser.add_argument('--integrator',
                    choices=['symplectic', 'runge-kutta-4'],
                    default='symplectic',
                    help='Numerical integration method for solving the spin dynamics')

parser.add_argument('--approximation',
                    choices=['classical-limit', 'quantum-approximation-sympy',
                             'quantum-exact', 'exact-bz-two-spins'],
                    required=True,
                    help='Approximation scheme to use')

parser.add_argument('--spin',
                    type=float,
                    required=True,
                    help='Quantum spin value (should normally be an integer multiple of 1/2)')

parser.add_argument('--order',
                    type=int,
                    default=1,
                    help='Order of the approximation scheme (up to order 4 for, "quantum-exact" and "quantum-approximation-sympy" methods)')

parser.add_argument('--exchange',
                    type=float,
                    required=True,
                    help='Exchange constant J value (in units of g muB)')

parser.add_argument('--from_difference',
                    type=bool,
                    default=True,
                    help='Computing effective Hamiltonian from difference to classical limit')

args = parser.parse_args()

integrator = args.integrator
qs = args.spin
approximation = args.approximation
J_factor = args.exchange
order = args.order
from_difference = args.from_difference


def main():
    alpha = 0.5  # Gilbert Damping parameter.

    b_z = 1.0  # Field z-component in Tesla
    applied_field = np.array((0, 0, b_z))

    # Temperature parameters
    temperatures = np.linspace(0.01, 10, 100)
    num_realisation = 5

    # Initial conditions
    s0 = np.array([
                   [1 / np.sqrt(3), 1.0 / np.sqrt(3), -1.0 / np.sqrt(3)]
                   ,
                   [1 / np.sqrt(3), 1.0 / np.sqrt(3), -1.0 / np.sqrt(3)]
                   ])  # Initial spin

    # Equilibration time, final time and time step
    equilibration_time = 5  # Equilibration time ns
    production_time = 15  # Final time ns
    time_step = 0.00005  # Time step ns, "linspace" so needs to turn num into int

    solver = asd.solver_factory(integrator, approximation, order, qs, applied_field, alpha, time_step, J_factor, from_difference)
    sz1 = asd.compute_temperature_dependence(solver, temperatures, time_step,
                                             equilibration_time, production_time, num_realisation, s0)

    file_name_1 = f'../data/qsd_{integrator}_{approximation}_{qs:.1f}_J={J_factor:.1f}.txt'

    header = f'spin: {qs}\n' \
             f'alpha: {alpha}\n' \
             f's0: {s0}\n' \
             f'integrator: {integrator}\n' \
             f'approximation: {approximation}\n' \
             f'time_step: {time_step}\n' \
             f'equilibration_time: {equilibration_time}\n' \
             f'production_time: {production_time}\n' \
             f'num_realisation: {num_realisation}\n' \
             f'\n' \
             'temperature_kelvin sz'

    if (approximation == 'quantum-approximation-sympy'
            or approximation == 'quantum-exact' or approximation == 'exact-bz-two-spins'):
        normalisation_factor = ((qs + 1)/qs)
    else:
        normalisation_factor = 1

    np.savetxt(file_name_1, np.column_stack((temperatures, normalisation_factor * sz1)), fmt='%.8e', header=header)


if __name__ == "__main__":
    start = time.process_time()
    main()
    end = time.process_time()
    print(f'runtime: {end-start:.3f} (s)')

import numpy as np
from scipy import constants as scp
from numba import njit
import effective_field_computation as efc

muB = scp.value("Bohr magneton")  # J T^-1
g_factor = np.fabs(scp.value("electron g factor"))  # dimensionless
gyro = scp.value("electron gyromag. ratio") * 1e-9  # in rad GHz T^-1
kB = scp.k  # J K^-1
g_muB_by_kB = g_factor * muB / kB


@njit
def rescale_spin(spin):
    """Returns normalised spin for a single spin"""
    return spin / np.linalg.norm(spin)


@njit
def rescale_spins(spins):
    """Returns normalised spins for two spin system"""
    for i in range(2):
        spins[i] = spins[i] / np.linalg.norm(spins[i])

    return spins


@njit
def random_field(time_step, temperature, alpha, quantum_spin):
    """Returns the stochastic thermal field which obeys the statistical properties

    〈ηᵢ(t)〉= 0
    〈ηᵢ(t) ηⱼ(t')〉= 2 α δᵢⱼ δ(t−t') / (β g μ_B s γ)

    corresponding to a classical white noise.

    Note the time step appears on the bottom below because we multiply by the time step
    in the numerical integration, so the overall effect is √(Δt) for the Wiener process.
    """
    gamma = np.sqrt((2 * kB * alpha * temperature)
                    / (time_step * g_factor * muB * quantum_spin * gyro))
    return np.random.normal(0, gamma, 3)


# Effective field computation
@njit
def effective_field_classical(spins, indice, applied_field, temperature, quantum_spin, J_factor):
    """Returns the classical effective field which for the Zeeman Hamiltonian is simply the
    effective field
    """
    return applied_field + J_factor * quantum_spin * spins[1-indice]


@njit
def classical_hamiltonian(spins, applied_field, quantum_spin, J_factor):
    """Returns the classical Hamiltonian"""
    return -(J_factor*g_factor*muB*quantum_spin**2*(spins[0][0]*spins[1][0] + spins[0][1]*spins[1][1] + spins[0][2]*spins[1][2])
             + g_factor*muB*quantum_spin*(applied_field[0]*(spins[0][0]+spins[1][0]) + applied_field[1]*(spins[0][1]+spins[1][1])
                                          + applied_field[2]*(spins[0][2]+spins[1][2])))


@njit
def rhs_runge_kutta_4(spins, indice, field, noise, alpha):
    """Returns the RHS of Landau-Lifshitz-Gilbert equation for RK4 integration"""
    torque = np.cross(spins[indice], field + noise)
    damping = np.cross(spins[indice], torque)

    rhs = -(gyro / (1 + alpha ** 2)) * (torque + alpha * damping)

    return rhs


@njit
def spin_advance_runge_kutta_4(spins, indice, field, time_step, temperature, alpha, quantum_spin, J_factor):
    """Given an initial spin s(t), returns s(t+dt) by RK4 integration
     of the damped precession around the effective field
     """
    noise = random_field(time_step, temperature, alpha, quantum_spin)
    spins_inter = np.copy(spins)

    rk_step_1 = rhs_runge_kutta_4(spins_inter, indice, field(spins_inter, indice, temperature, quantum_spin, J_factor)
                                  , noise, alpha)
    spins_inter[indice] = rescale_spin(spins[indice] + (time_step / 2) * rk_step_1)

    rk_step_2 = rhs_runge_kutta_4(spins_inter, indice, field(spins_inter, indice, temperature, quantum_spin, J_factor)
                                  , noise, alpha)
    spins_inter[indice] = rescale_spin(spins[indice] + (time_step / 2) * rk_step_2)

    rk_step_3 = rhs_runge_kutta_4(spins_inter, indice, field(spins_inter, indice, temperature, quantum_spin, J_factor)
                                  , noise, alpha)
    spins_inter[indice] = rescale_spin(spins[indice] + time_step * rk_step_3)

    rk_step_4 = rhs_runge_kutta_4(spins_inter, indice, field(spins_inter, indice, temperature, quantum_spin, J_factor)
                                  , noise, alpha)

    new_spin = spins[indice] + time_step * (rk_step_1 + 2 * rk_step_2 + 2 * rk_step_3 + rk_step_4) / 6

    return rescale_spin(new_spin)


@njit
def advance_system_runge_kutta_4(spins, field, time_step, temperature, alpha, quantum_spin, J_factor):
    """Advances the whole spin system from time t to time t+dt by Runge-Kutta 4 integration
         of the damped precession around the effective field
         """
    spins_inter = spin_advance_runge_kutta_4(spins, 0, field, time_step, temperature
                                             , alpha, quantum_spin, J_factor)
    spins[1] = spin_advance_runge_kutta_4(spins, 1, field, time_step, temperature
                                          , alpha, quantum_spin, J_factor)
    spins[0] = spins_inter
    return spins


@njit
def spin_advance_symplectic(spins, indice, field, time_step, temperature, alpha, quantum_spin, J_factor):
    """Given an initial spin s(t), returns s(t+dt) by symplectic integration
     of the damped precession around the effective field
     """
    effective_field = field(spins, indice, temperature, quantum_spin, J_factor) \
                      + random_field(time_step, temperature, alpha, quantum_spin)

    effective_precession = (gyro / (1 + alpha ** 2)) \
                           * (effective_field + alpha * np.cross(spins[indice], effective_field))

    torque = np.cross(effective_precession, spins[indice])
    energy = np.dot(effective_precession, spins[indice])

    precession_norm = np.linalg.norm(effective_precession)

    norm_by_timestep = precession_norm * time_step
    energy_over_norm = energy / precession_norm
    cos_precession = np.cos(norm_by_timestep)
    sin_precession = np.sin(norm_by_timestep)

    return cos_precession * spins[indice] + ((sin_precession * torque) + energy_over_norm * (
                1.0 - cos_precession) * effective_precession) / precession_norm


@njit
def advance_system_symplectic(spins, field, time_step, temperature, alpha, quantum_spin, J_factor):
    """Advances the whole spin system from time t to time t+dt by Symplectic integration"""
    spins[0] = spin_advance_symplectic(spins, 0, field, time_step / 2, temperature, alpha, quantum_spin, J_factor)
    spins[1] = spin_advance_symplectic(spins, 1, field, time_step, temperature, alpha, quantum_spin, J_factor)
    spins[0] = spin_advance_symplectic(spins, 0, field, time_step / 2, temperature, alpha, quantum_spin, J_factor)
    return spins


def solver_factory(method, approximation, order, quantum_spin, applied_field, alpha, time_step, J_factor, from_difference = True):
    """Returns the atomistic solver corresponding to the method of integration and approximation
    for the computation of the effective field
    """
    if approximation == 'classical-limit':
        @njit
        def field_function(spins, indice, temperature, quantum_spin, J_factor):
            return effective_field_classical(spins, indice, applied_field, temperature, quantum_spin, J_factor)
    elif approximation == 'quantum-approximation-sympy':
        field_from_hamiltonian = efc.numerical_field(quantum_spin, order, from_difference)

        @njit
        def field_function(spins, indice, temperature, quantum_spin, J_factor):
            return field_from_hamiltonian(spins[indice], spins[1 - indice], applied_field, g_factor,
                                          J_factor * g_factor * muB, muB, kB, temperature)
    elif approximation == 'quantum-exact':
        field_from_hamiltonian = efc.numerical_field_exact(quantum_spin, order, from_difference)

        @njit
        def field_function(spins, indice, temperature, quantum_spin, J_factor):
            return field_from_hamiltonian(spins[indice], spins[1 - indice], applied_field, g_factor,
                                          J_factor * g_factor * muB, muB, kB, temperature)

    elif approximation == 'exact-bz-two-spins':
        field_from_hamiltonian = efc.numerical_field_exact_bz_two_spins(quantum_spin)

        @njit
        def field_function(spins, indice, temperature, quantum_spin, J_factor):
            return field_from_hamiltonian(spins[indice], spins[1 - indice], applied_field, g_factor,
                                          J_factor * g_factor * muB, muB, kB, temperature)
    else:
        raise RuntimeError(f'Unknown approximation: {approximation}')

    if method == 'symplectic':
        @njit
        def solver_function(spins, temperature):
            return advance_system_symplectic(
                spins, field_function, time_step, temperature, alpha, quantum_spin, J_factor)
    elif method == 'runge-kutta-4':
        @njit
        def solver_function(spins, temperature):
            return advance_system_runge_kutta_4(
                spins, field_function, time_step, temperature, alpha, quantum_spin, J_factor)
    else:
        raise RuntimeError(f'Unknown integrator: {method}')

    return solver_function


# Result computation
@njit
def calculate_sz_asd(solver, spins_initial, temperature, num_eq_steps, num_production_steps,
                     num_realisations):
    """Returns the value of the expectation value of the z-component of the spin by averaging over
    time and realisations of the noise
    """
    sz_realisations = 0.0
    for _ in range(num_realisations):
        # Incase the initial spin is not properly normalised
        spins = rescale_spins(spins_initial)

        for _ in range(0, num_eq_steps):
            spins = solver(spins, temperature)

        spins_z = 0.0
        for _ in range(0, num_production_steps):
            spins = solver(spins, temperature)
            spins_z += spins[0][2] + spins[1][2]

        sz_realisations += spins_z / (2 * num_production_steps)

    return sz_realisations / num_realisations


@njit
def calculate_H_classical_asd(solver, spins_initial, temperature, num_eq_steps, num_production_steps,
                     num_realisations, applied_field, quantum_spin, J_factor):
    """Returns the value of the expectation value of the z-component of the spin by averaging over
    time and realisations of the noise
    """

    h_realisations = 0.0
    for _ in range(num_realisations):
        # Incase the initial spin is not properly normalised
        spins = rescale_spins(spins_initial)

        for _ in range(0, num_eq_steps):
            spins = solver(spins, temperature)

        h_classical = 0.0
        for _ in range(0, num_production_steps):
            spins = solver(spins, temperature)
            h_classical += classical_hamiltonian(spins, applied_field, quantum_spin, J_factor)

        h_realisations += h_classical / num_production_steps

    return h_realisations / num_realisations

@njit
def compute_temperature_dependence(solver, temperatures, time_step,
                                   equilibration_time, production_time, num_realisation,
                                   spins_initial):
    """Returns an array of expectation values of the z-component of the spin corresponding to the
    input temperatures"""
    sz_expectation = np.zeros(np.shape(temperatures))

    for i in range(len(temperatures)):
        sz_expectation[i] = calculate_sz_asd(solver, spins_initial, temperatures[i],
                                             int(equilibration_time / time_step)
                                             , int(production_time / time_step),
                                             num_realisation)

    return sz_expectation


@njit
def compute_temperature_dependence_hamiltonian_classical(solver, temperatures, time_step,
                                   equilibration_time, production_time, num_realisation,
                                   spins_initial, applied_field, quantum_spin, J_factor):
    """Returns an array of expectation values of the z-component of the spin corresponding to the
    input temperatures"""
    h_classical_expectation = np.zeros(np.shape(temperatures))

    for i in range(len(temperatures)):
        h_classical_expectation[i] = calculate_H_classical_asd(solver, spins_initial, temperatures[i],
                                             int(equilibration_time / time_step)
                                             , int(production_time / time_step),
                                             num_realisation, applied_field, quantum_spin, J_factor)

    return h_classical_expectation

from sympy import *
from numba import njit
import numpy as np
from sympy.physics.quantum.cg import CG
from sympy import exp
from fractions import Fraction
import generating_moments as gm

a = Symbol('s1x', commutative=False)
b = Symbol('s1y', commutative=False)
c = Symbol('s1z', commutative=False)

A = Symbol('s1up', commutative=False)
B = Symbol('s1down', commutative=False)
C = Symbol('s2up', commutative=False)
D = Symbol('s2down', commutative=False)

E = (A + B) / 2
F = (A - B) / (2 * I)
G = (C + D) / 2
H = (C - D) / (2 * I)

d = Symbol('s2x', commutative=False)
e = Symbol('s2y', commutative=False)
f = Symbol('s2z', commutative=False)

g = Symbol('nx1', commutative=True, real=True)
h = Symbol('ny1', commutative=True, real=True)
i = Symbol('nz1', commutative=True, real=True)

j = Symbol('nx2', commutative=True, real=True)
k = Symbol('ny2', commutative=True, real=True)
l = Symbol('nz2', commutative=True, real=True)

m = Symbol('Bx', commutative=True, real=True)
n = Symbol('By', commutative=True, real=True)
o = Symbol('Bz', commutative=True, real=True)

p = Symbol('g', commutative=True, real=True)
q = Symbol('J', commutative=True, real=True)
r = Symbol('muB', commutative=True, real=True)

P = Symbol('gmuB', commutative=True, real=True)

s = Symbol('S', commutative=True, real=True)
S = Symbol('doubleS', real=True, positive=True, integer=True)

t = Symbol('beta', commutative=True, real=True)
u = Symbol('k_B', commutative=True, real=True)
v = Symbol('T', commutative=True, real=True)

x = Symbol('z', commutative=True)
y = Symbol('z_star', commutative=True)
z = Symbol('|z|', commutative=True)

# Auxiliary variables
z1 = Symbol('z1', complex=True)
z2 = Symbol('z2', complex=True)

L = Symbol('L', real=True)
J = Symbol('J', real=True, nonnegative=True, integer=True)
K = Symbol('K', real=True, nonnegative=True)

s1 = Matrix([a, b, c])
s2 = Matrix([d, e, f])

n1 = Matrix([g, h, i])
n2 = Matrix([j, k, l])
B_field = Matrix([m, n, o])


def overlaps(total_spin, spin_z_1, spin_z_2, sum_spin, spin_z):
    """Returns the Clebsch-Gordan coefficients needed to compute overlaps with spin coherent states"""
    cg = CG(total_spin, spin_z_1, total_spin, spin_z_2, sum_spin, spin_z)
    res = cg.doit()
    return res


def scselem(total_spin, z1_power, z2_power):
    """Returns the spin coherent state matrix element:
     <z1^z1_power*z2^z2_power>
     """
    return sqrt(binomial(int(2 * total_spin), z1_power) * binomial(int(2 * total_spin), z2_power))*z1**z1_power*z2**z2_power


def spin_coherent_state_overlap(total_spin, sum_spin, spin_z):
    """Returns the square modulus of the overlap between the diagonalised spin system with the spin coherent states"""
    res = 0
    for z1_power in range(0, int(2 * total_spin + 1)):
        for z2_power in range(0, int(2 * total_spin + 1)):
            res = res + scselem(total_spin, z1_power, z2_power) * overlaps(total_spin, total_spin - z1_power, total_spin - z2_power, sum_spin, spin_z)
    return Abs(res)**2


def substitute_z(function):
    """Replaces z_1 and z_2 coefficient from spin coherent state with corresponding unit spin coherent state vector components"""
    result = function.subs(z1, (g+I*h)*(1+(Abs(z1))**2)/2)
    result = result.subs(z2, (j+I*k)*(1+(Abs(z2))**2)/2)
    return result


def substitute_module_z_squared(function):
    """Replaces |z_1|^2 and |z_2|^2 coefficient from spin coherent state with corresponding unit spin coherent state vector components"""
    result = function.subs(Abs(z1)**2, (1-i)/(1+i))
    result = result.subs(Abs(z2)**2, (1-l)/(1+l))
    return result


def denomin(total_spin):
    """Returns the appropriate normalisation denominator for the spin coherent state
    with the unit spin coherent state vector components
    """
    res = ((1+Abs(z1)**2)*(1+Abs(z2)**2))**(2 * total_spin)
    res = substitute_module_z_squared(res)
    return simplify(res)


def diagonal_hamiltonian(total_spin, Spin, Mz):
    """Returns the properly diagonalised hamiltonian in terms of the corresponding single spin"""
    function = -q/2*(Spin*(Spin+1)-2*(total_spin+1)*total_spin)-p*r*o*Mz
    return function


def exact_matrix_elem(total_spin):
    """Returns the exact matrix element using the Clebsch-Gordan coefficients and the overlaps with
    the spin coherent states in terms of the unit spin coherent state vector
    """
    function = 0
    counter = 0
    eigen = eigenvalues(total_spin)
    counter2 = 0
    for eig in eigen:
        eigen[counter2] = exp(-t * eig)
        counter2 = counter2 + 1
    for Sp in range(0, int(2*total_spin+1)):
        for M in range(-Sp, Sp+1):
            result = factor(
                substitute_module_z_squared(expand(substitute_z(spin_coherent_state_overlap(total_spin, Sp, M))))).subs(
                (1 + i), i + Fraction(1, 2) * (g ** 2 + h ** 2 + i ** 2 + j ** 2 + k ** 2 + l ** 2)).subs((1 + l),
                                                                                                          l + Fraction(
                                                                                                              1, 2) * (
                                                                                                                      g ** 2 + h ** 2 + i ** 2 + j ** 2 + k ** 2 + l ** 2))
            denominator = denomin(total_spin).subs((1 + i), i + Fraction(1, 2) * (
                        g ** 2 + h ** 2 + i ** 2 + j ** 2 + k ** 2 + l ** 2)).subs((1 + l), l + Fraction(1, 2) * (
                        g ** 2 + h ** 2 + i ** 2 + j ** 2 + k ** 2 + l ** 2))
            result2 = result / denominator
            function = function + result2 * eigen[counter]
            counter = counter + 1
    return function


def eigenvalues(Spin):
    """Returns the eigenvalues of diagonalised Hamiltonian"""
    eigenvalues = Matrix.zeros((int(2*Spin+1))**2, 1)
    counter = 0
    for spin in range(0,int(2*Spin+1)):
        for mz in range(-spin,spin+1):
            eigenvalues[counter] = diagonal_hamiltonian(Spin,spin,mz)
            counter = counter+1
    return eigenvalues


def quantum_thermal_exponent():
    """Returns the quantum Hamiltonian for the partition function"""
    return q * s1.dot(s2) + p * r * B_field.dot(s1 + s2)


def classical_thermal_exponent():
    """Returns the Classical Hamiltonian for the partition function"""
    return q * s**2 * n1.dot(n2) + p * r * s * B_field.dot(n1 + n2)


def quantum_thermal_exponent_from_difference():
    """Returns the quantum Hamiltonian for the partition function computed from the difference to the classical limit"""
    return q * s1.dot(s2) + p * r * B_field.dot(s1 + s2) - (q * s**2 * n1.dot(n2) + p * r * s * B_field.dot(n1 + n2))


def exponential_series(function, order, parameter):
    """Returns the exponential series for the given order and parameter (usually 1/(k_B*T))"""
    result = 1
    for ind in range(1, order + 1):
        result += (function ** ind) * (parameter ** ind) * Rational(1, factorial(ind))
    return result


def compute_hamiltonian(order):
    """Returns the quantum Hamiltonian for the given order"""
    q_thermal_exponent = quantum_thermal_exponent()
    quantum_series = exponential_series(q_thermal_exponent, order, t)
    hamiltonian = expand(quantum_series)
    return hamiltonian


def compute_hamilton_exact_bz_two_spins(quantum_spin):
    """Returns the exact Hamiltonian for a given quantum spin number (field only along z for two spins)"""
    return ln(exact_matrix_elem(quantum_spin))


def compute_hamiltonian_from_difference(order):
    """Returns the quantum Hamiltonian as a difference to the classical limit, for the given order"""
    q_thermal_exponent = quantum_thermal_exponent_from_difference()
    quantum_series = exponential_series(q_thermal_exponent, order, t)
    hamiltonian = expand(quantum_series)
    return hamiltonian


def replace_moments_any_order(order, function, moments_all):
    """Returns the expression of function with all the moments in terms of complex z replaced
    by corresponding spin coherent state unit vector components
    """
    moments_1 = gm.compute_moments_real(S, s, order, E, F, c, A, B, x, y, z, g, h, i, L, J, K, moments_all)
    moments_2 = gm.compute_moments_real(S, s, order, G, H, f, C, D, x, y, z, j, k, l, L, J, K, moments_all)
    print(moments_1)
    print(moments_2)

    size = binomial(order + 2, 2)
    spin_op_list_1 = MutableDenseNDimArray(np.zeros(size), (size,))
    spin_op_list_2 = MutableDenseNDimArray(np.zeros(size), (size,))
    arrays = gm.generate_and_sort_arrays_with_sum(order)
    spin_elements_1 = np.array(([a, b, c]))
    spin_elements_2 = np.array(([d, e, f]))
    elem = 0
    moments_replaced = function
    for array in arrays:
        result = 1
        for depth in range(3):
            result = result * spin_elements_1[depth] ** array[depth]
        spin_op_list_1[elem] = result
        elem = elem + 1
    elem = 0
    for array in arrays:
        result = 1
        for depth in range(3):
            result = result * spin_elements_2[depth] ** array[depth]
        spin_op_list_2[elem] = result
        elem = elem + 1
    for depth in range(size):
        final_result = expand(moments_replaced).subs(spin_op_list_1[depth], moments_1[depth])
        moments_replaced = final_result
    for depth in range(size):
        final_result = expand(moments_replaced).subs(spin_op_list_2[depth], moments_2[depth])
        moments_replaced = final_result
    return moments_replaced


def hamiltonian_from_taylor(function, order):
    """Returns an approximated quantum Hamiltonian using a Taylor series up to given order"""
    first_step = series(ln(simplify(function)), t, 0, order + 1, dir='+').removeO()
    effective_hamiltonian = first_step.subs(t, 1 / (u * v))
    return effective_hamiltonian


def classicalisation(order, from_difference):
    """Returns the classical high temperature approximation of the quantum Hamiltonian up to a given order"""
    if from_difference:
        hamiltonian = compute_hamiltonian_from_difference(order)
    else:
        hamiltonian = compute_hamiltonian(order)
    hamiltonian = separate_site_operators_any_order(order, hamiltonian)
    moments_all = 1
    print("moments generated")
    for ind in reversed(range(1, order + 1)):
        hamiltonian = replace_with_commutators_any_order(ind, hamiltonian)

    for ind in reversed(range(1, order + 1)):
        hamiltonian = replace_moments_any_order(ind, hamiltonian, moments_all)

    hamiltonian = simplify(hamiltonian.subs(j ** 2, 1 - k ** 2 - l ** 2).subs(g ** 2, 1 - h ** 2 - i ** 2))
    hamiltonian = simplify(hamiltonian_from_taylor(hamiltonian, order))
    print("final form of hamiltonian obtained")
    return hamiltonian


def generate_effective_field_for_c(order, from_difference):
    """Prints the expression of the effective field obtained from the
    classicalisation method to be used in a separate C code
    """
    hamiltonian = classicalisation(order, from_difference)
    eff_field = np.squeeze(simplify(1 / (t * P * s) * diff(hamiltonian, n1)))
    if from_difference:
        eff_field = simplify(
            eff_field + np.squeeze(1 / (P * s) * diff(classical_thermal_exponent(), n1)))
    eff_field_2 = np.array([
        simplify(eff_field[0].subs(j, 0).subs(k, 0).subs(l, 0).subs(q, 0)),
        simplify(eff_field[1].subs(j, 0).subs(k, 0).subs(l, 0).subs(q, 0)),
        simplify(eff_field[2].subs(j, 0).subs(k, 0).subs(l, 0).subs(q, 0))
                            ])
    eff_field_3 = simplify(eff_field - eff_field_2)
    print("h_i[0] = ", ccode(eff_field_2[0]), ";")
    print("h_i[1] = ", ccode(eff_field_2[1]), ";")
    print("h_i[2] = ", ccode(eff_field_2[2]), ";")
    print("h_i[0] += ", ccode(eff_field_3[0]), ";")
    print("h_i[1] += ", ccode(eff_field_3[1]), ";")
    print("h_i[2] += ", ccode(eff_field_3[2]), ";")
    return eff_field


def generate_effective_field_for_c_exact(order, from_difference):
    """Prints the expression of the effective field obtained from the
    classicalisation_exact method to be used in a separate C code
    """
    hamiltonian = classicalisation_exact(order, from_difference)
    eff_field = np.squeeze(simplify(1 / (t * P * s) * diff(hamiltonian, n1)))
    if from_difference:
        eff_field = simplify(
            eff_field + np.squeeze(1 / (P * s) * diff(classical_thermal_exponent(), n1)))
    eff_field_2 = np.array([
        simplify(eff_field[0].subs(j, 0).subs(k, 0).subs(l, 0).subs(q, 0)),
        simplify(eff_field[1].subs(j, 0).subs(k, 0).subs(l, 0).subs(q, 0)),
        simplify(eff_field[2].subs(j, 0).subs(k, 0).subs(l, 0).subs(q, 0))
                            ])
    eff_field_3 = eff_field - eff_field_2
    print("h_i[0] = ", ccode(eff_field_2[0]), ";")
    print("h_i[1] = ", ccode(eff_field_2[1]), ";")
    print("h_i[2] = ", ccode(eff_field_2[2]), ";")
    print("h_i[0] += ", ccode(eff_field_3[0]), ";")
    print("h_i[1] += ", ccode(eff_field_3[1]), ";")
    print("h_i[2] += ", ccode(eff_field_3[2]), ";")
    return eff_field


def effective_field(quantum_spin, order, from_difference):
    """Returns the effective field obtained from the classicalisation method hamiltonian"""
    hamiltonian = classicalisation(order, from_difference)
    as_function_of_quantum_spin = simplify(hamiltonian.subs(s, quantum_spin))
    eff_field = simplify(u * v / (p * r * quantum_spin) * diff(as_function_of_quantum_spin, n1))
    if from_difference:
        eff_field = simplify(
            eff_field + 1 / (p * r * quantum_spin) * diff(classical_thermal_exponent().subs(s, quantum_spin), n1))
        return eff_field
    else:
        return eff_field


def effective_field_exact_bz_two_spins(quantum_spin):
    """Returns the effective field obtained from the compute_hamilton_exact_bz_two_spins method"""
    hamiltonian = compute_hamilton_exact_bz_two_spins(quantum_spin)
    print("effective hamiltonian obtained")
    log_dif = diff(hamiltonian, n1).subs((i+1)*(l+1), (i+g**2+h**2+i**2)*(l+j**2+k**2+l**2))
    print("log form differentiated")
    eff_field = (u * v) / (p * r * quantum_spin) * log_dif
    print("effective field obtained")
    return eff_field


def numerical_field_exact_bz_two_spins(quantum_spin):
    """Returns a njit numerically usable field function for the effective_field_exact_bz_two_spins method"""
    generate_effective_field_for_c_exact_bz_two_spins(quantum_spin)
    # input()
    inter = effective_field_exact_bz_two_spins(quantum_spin).subs(t, 1/(u*v))
    inter2 = Array(([inter[0], inter[1], inter[2]]))

    func = lambdify([n1, n2, B_field, p, q, r, u, v], inter2, 'numpy')
    print("numerical field function generated")
    return njit(func)


def generate_effective_field_for_c_exact_bz_two_spins(quantum_spin):
    """Prints the expression of the effective field obtained from the
    compute_hamilton_exact_bz_two_spins method to be used in a separate C code
    """
    hamiltonian = compute_hamilton_exact_bz_two_spins(quantum_spin)
    eff_field_start = diff(hamiltonian, n1).subs((i+1)*(l+1), (i+g**2+h**2+i**2)*(l+j**2+k**2+l**2))
    eff_field = np.squeeze(1 / (t * P * quantum_spin) * eff_field_start.subs(p*r, P))
    print("h_i[0] = ", ccode(eff_field[0]), ";")
    print("h_i[1] = ", ccode(eff_field[1]), ";")
    print("h_i[2] = ", ccode(eff_field[2]), ";")
    return eff_field


def numerical_field(quantum_spin, order, from_difference):
    """Returns a njit numerically usable field function for the effective_field method"""
    # generate_effective_field_for_c(order, from_difference)
    # input()
    inter = effective_field(quantum_spin, order, from_difference)
    inter2 = Array(([inter[0], inter[1], inter[2]]))

    func = lambdify([n1, n2, B_field, p, q, r, u, v], inter2, 'numpy')
    print("numerical field function generated")
    return njit(func)


def hamiltonian_exact(function, order):
    """Returns logarithm of the provided function replacing the parameter t [beta] by 1/(u*v) [1/(k_B*T)]
    for the exact classical Hamiltonian computation
    """
    first_step = ln(simplify(function))
    effective_hamiltonian = first_step.subs(t, 1 / (u * v))
    return effective_hamiltonian


def classicalisation_exact(order, from_difference):
    """Returns the "exact" effective classical Hamiltonian (i.e. logarithm expression),
    potentially computed from the difference to the classical limit, if from_difference is set to True
    """
    if from_difference:
        hamiltonian = compute_hamiltonian_from_difference(order)
    else:
        hamiltonian = compute_hamiltonian(order)
    hamiltonian = separate_site_operators_any_order(order, hamiltonian)
    moments_all = 1
    for ind in reversed(range(1, order + 1)):
        hamiltonian = replace_with_commutators_any_order(ind, hamiltonian)

    for ind in reversed(range(1, order + 1)):
        hamiltonian = replace_moments_any_order(ind, hamiltonian, moments_all)

    hamiltonian = simplify(hamiltonian.subs(j ** 2, 1 - k ** 2 - l ** 2).subs(g ** 2, 1 - h ** 2 - i ** 2))
    hamiltonian = simplify(hamiltonian_exact(hamiltonian, order))
    print("final form of hamiltonian obtained")
    return hamiltonian


def effective_field_exact(quantum_spin, order, from_difference):
    """Returns the effective field computed from classicalisation_exact method"""
    hamiltonian = classicalisation_exact(order, from_difference)
    as_function_of_quantum_spin = simplify(hamiltonian.subs(s, quantum_spin))
    eff_field = simplify(u * v / (p * r * quantum_spin) * diff(as_function_of_quantum_spin, n1))
    if from_difference:
        eff_field = simplify(eff_field + 1 / (p * r * quantum_spin) * diff(classical_thermal_exponent().subs(s, quantum_spin),n1))
        return eff_field
    else:
        return eff_field


def numerical_field_exact(quantum_spin, order, from_difference):
    """Returns a njit numerically usable field function for the effective_field_exact method"""
    # generate_effective_field_for_c_exact(order, from_difference)
    # input()
    inter = effective_field_exact(quantum_spin, order, from_difference)
    inter2 = Array(([inter[0], inter[1], inter[2]]))
    func = lambdify([n1, n2, B_field, p, q, r, u, v], inter2, 'numpy')
    print("numerical field function generated")
    return njit(func)


def separate_site_operators_any_order(order, function):
    """Returns an ordered expression of the provided moments where operators acting on site 1 are on the left side
    and those acting on site 2 are on the right side
    """
    site_separated = function
    for _ in range(binomial(order + 1, 2)):
        intermediate = expand(site_separated).subs(d * a, a * d).subs(d * b, b * d).subs(d * c, c * d).subs(e * a,
                                a * e).subs(e * b, b * e).subs(e * c, c * e).subs(f * a, a * f).subs(f * b,
                                b * f).subs(f * c, c * f)
        site_separated = intermediate
    return site_separated


def replace_with_commutators_any_order(order, function):
    """Orders all the matrix elements in function in such a way that they can all be replaced with the computed
    set from generating_moments.py
    """
    ordered_expression = function
    for _ in range(binomial(order + 1, 2)):
        intermediate_1 = expand(ordered_expression).subs(b * a, a * b - I * c).subs(c * b, b * c - I * a).subs(c * a, a * c + I * b)
        intermediate_2 = expand(intermediate_1).subs(e * d, d * e - I * f).subs(f * e, e * f - I * d).subs(f * d,
                                                                                                     d * f + I * e)
        ordered_expression = intermediate_2
    return ordered_expression


def main():
    # numerical_field_exact(1, 2, False)
    numerical_field_exact_bz_two_spins(1/2)
    # moments = gm.compute_moments_list_complex_any_order(S, s, L, x, J, K, z, y, 3)
    # for moment in moments:
    #     print_latex(collect(moment,S))
    #     print("\\\\&")



if __name__ == "__main__":
    main()

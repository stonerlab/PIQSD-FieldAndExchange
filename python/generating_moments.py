from sympy import *
import numpy as np


def application_s_minus(S, y, p):
    """Returns the action of (S^-)^y on |p>"""
    res = 1
    for i in range(y):
        res = res * (S-(p+i))*((p+i)+1)
    return res


def application_s_plus_from_minus(S, x, p, y):
    """Returns the action of (S^+)^x on (S^-)^y|p>"""
    res = 1
    for i in range(x):
        res = res * ((p+y)-i)*(S-((p+y)-i)+1)
    return res


def correct_square_root(S, x, y, p, q):
    """Returns the proper prefactor for the spin coherent state <q|(S^-)^x(S^+)^y|p> depending on if x>y or not"""
    if y >= x:
        res = application_s_minus(S, y, p) * application_s_plus_from_minus(S, x, p, y) * binomial(S,p) * binomial(S,p-x+y)
    else:
        res = application_s_minus(S, y, q) * application_s_plus_from_minus(S, x, q, y) * binomial(S,p) * binomial(S,p+x-y)
    return res


def sum_multiplier(S, x, y, p, q):
    """Returns the properly normalised expression of the prefactor for <q|(S^-)^x(S^+)^y|p>"""
    if y >= x:
        res = correct_square_root(S, x, y, p, q)
        res2 = simplify(res/(binomial(S-y,p)**2))
    else:
        res = correct_square_root(S, x, y, p, q).subs(q,p+x-y)
        res2 = simplify(res/(binomial(S-x,p)**2))
    return sqrt(res2)


def matrix_element_plus_minus_z(S, L, z, J, K, x, y, v):
    """Returns the matrix element <J|(S^-)^x(S^+)^y(S^z)^v|K>"""
    if y >= x:
        res = binomial(S - y, J) * sum_multiplier(S, x, y, J, K) * (L ** K) * (z ** J) * KroneckerDelta(K, J - x + y)
        res = res * (S / 2.0 - J) ** v
    else:
        res = binomial(S - x, J) * sum_multiplier(S, x, y, J, K) * (L ** J) * (z ** K) * KroneckerDelta(K, J + x - y)
        res = res * (S / 2.0 - (J + x - y)) ** v
    return expand(res)


def matrix_elements_spin_plus_minus_z(S, s, L, z, J, K, u, w, x, y, v):
    """Returns the matrix element <z|(S^-)^x(S^+)^y(S^z)^v|z>"""
    res = matrix_element_plus_minus_z(S, L, z, J, K, x, y, v)
    if y >= x:
        res2 = summation(summation(res, (K, 0, S - y)).args[0].args[0], (J, 0, S - y))
    else:
        res2 = summation(summation(res, (K, 0, S - x)).args[0].args[0], (J, 0, S - x))
    res3 = 0

    if (x+y+v) % 2 != 0 or v == 0 or v % 2 != 0 or (v % 2 == 0 and (x+y) % 2 ==0 and x+y != 0) :
        for elem in range(int(len(res2.args))):
            factor = 1
            for depth in range(len(res2.args[elem].args)):
                if depth == len(res2.args[elem].args) - 1:
                    factor = factor * simplify(res2.args[elem].args[depth]).args[0].args[0]
                else:
                    factor = factor * res2.args[elem].args[depth]
            res3 = res3 + factor
    else:
        for elem in range(int(len(res2.args))):
            factor = 1
            for depth in range(len(res2.args[elem].args)):
                if depth == len(res2.args[elem].args) - 1:
                    if elem != int(len(res2.args) - 1):
                        factor = factor * simplify(res2.args[elem].args[depth]).args[0].args[0]
                else:
                    if elem == len(res2.args) - 1:
                        factor = factor * simplify(res2.args[elem].args[depth])[0]
                    else:
                        factor = factor * res2.args[elem].args[depth]
            res3 = res3 + factor
    res4 = res3/((1 + (L * z)) ** S)
    return simplify(res4.subs(L * z, u ** 2).subs(L, w).subs(S, 2 * s))


def compute_moments_list_complex_any_order(S, s, L, z, J, K, mod_z, z_star, order):
    """Returns all matrix elements of:
    <z|(S^-)^x(S^+)^y(S^z)^v|z>
    for the given order using matrix_elements_spin_plus_minus_z
    """
    if order != 1:
        size = binomial(order + 2, 2)
        moments = MutableDenseNDimArray(np.zeros(size), (size,))
        arrays = generate_and_sort_arrays_with_sum(order)
        elem = 0
        for array in arrays:
            moments[elem] = matrix_elements_spin_plus_minus_z(S, s, L, z, J, K, mod_z, z_star, array[0], array[1], array[2])
            moments[elem] = simplify(moments[elem].collect((1+mod_z**2)**s))
            for ind in range(2, order):
                if ind % 2 == 0:
                    exp = factor(2 * s - ind).args[1]
                else:
                    exp = factor(2 * s - ind)
                moments[elem] = simplify(moments[elem].subs(Abs(exp), (exp)))
            elem = elem + 1
        return moments
    else :
        moments = MutableDenseNDimArray(np.zeros(3), (3,))
        moments[0] = 2 * s * z / (1 + mod_z ** 2)
        moments[1] = 2 * s * z_star / (1 + mod_z ** 2)
        moments[2] = s * (1 - mod_z ** 2) / (1 + mod_z ** 2)
        return moments


def generate_moments_list_up_to_order(S, s, L, z, J, K, mod_z, z_star, order):
    """Returns all matrix elements of:
    <z|(S^-)^x(S^+)^y(S^z)^v|z>
    up to the given order using compute_moments_list_complex_any_order
    """
    moments = []
    for ind in range(1, order+1):
        moments.append(compute_moments_list_complex_any_order(S, s, L, z, J, K, mod_z, z_star, ind))
    return moments


def compute_moments_list_complex(S, s, L, z, J, K, mod_z, z_star, order):
    """Returns all matrix elements of:
    <z|(S^-)^x(S^+)^y(S^z)^v|z>
    up to the given order (5 max) using hard-coded results from generate_moments_list_up_to_order for efficiency
    """
    if order == 5:
        moments = MutableDenseNDimArray(np.zeros(21), (21,))
        moments[0] = 8 * s * z ** 5 * (2 * s - 1) * (2 * s ** 3 - 9 * s ** 2 + 13 * s - 6) / (
                    mod_z ** 10 + 5 * mod_z ** 8 + 10 * mod_z ** 6 + 10 * mod_z ** 4 + 5 * mod_z ** 2 + 1)
        moments[1] = 8 * s * z ** 3 * (2 * s ** 2 * mod_z ** 2 - s * mod_z ** 2 + 4 * s - 2) * (
                    2 * s ** 2 - 5 * s + 3) / (
                                 mod_z ** 10 + 5 * mod_z ** 8 + 10 * mod_z ** 6 + 10 * mod_z ** 4 + 5 * mod_z ** 2 + 1)
        moments[2] = s * z ** 4 * (-8 * s ** 2 * mod_z ** 2 + 8 * s ** 2 + 4 * s * mod_z ** 2 - 36 * s + 16) * (
                    2 * s ** 2 - 5 * s + 3) / (
                                 mod_z ** 10 + 5 * mod_z ** 8 + 10 * mod_z ** 6 + 10 * mod_z ** 4 + 5 * mod_z ** 2 + 1)
        moments[3] = 8 * s * z * (
                    4 * s ** 4 * mod_z ** 4 - 8 * s ** 3 * mod_z ** 4 + 12 * s ** 3 * mod_z ** 2 + 5 * s ** 2 * mod_z ** 4 - 24 * s ** 2 * mod_z ** 2 + 6 * s ** 2 - s * mod_z ** 4 + 15 * s * mod_z ** 2 - 9 * s - 3 * mod_z ** 2 + 3) / (
                                 mod_z ** 10 + 5 * mod_z ** 8 + 10 * mod_z ** 6 + 10 * mod_z ** 4 + 5 * mod_z ** 2 + 1)
        moments[4] = s * z ** 2 * (
                    -16 * s ** 4 * mod_z ** 4 + 16 * s ** 4 * mod_z ** 2 + 40 * s ** 3 * mod_z ** 4 - 96 * s ** 3 * mod_z ** 2 + 24 * s ** 3 - 32 * s ** 2 * mod_z ** 4 + 164 * s ** 2 * mod_z ** 2 - 84 * s ** 2 + 8 * s * mod_z ** 4 - 108 * s * mod_z ** 2 + 84 * s + 24 * mod_z ** 2 - 24) / (
                                 mod_z ** 10 + 5 * mod_z ** 8 + 10 * mod_z ** 6 + 10 * mod_z ** 4 + 5 * mod_z ** 2 + 1)
        moments[5] = s * z ** 3 * (
                    8 * s ** 4 * mod_z ** 4 - 16 * s ** 4 * mod_z ** 2 + 8 * s ** 4 - 12 * s ** 3 * mod_z ** 4 + 88 * s ** 3 * mod_z ** 2 - 60 * s ** 3 + 4 * s ** 2 * mod_z ** 4 - 128 * s ** 2 * mod_z ** 2 + 148 * s ** 2 + 68 * s * mod_z ** 2 - 132 * s - 12 * mod_z ** 2 + 36) / (
                                 mod_z ** 10 + 5 * mod_z ** 8 + 10 * mod_z ** 6 + 10 * mod_z ** 4 + 5 * mod_z ** 2 + 1)
        moments[6] = 8 * s * z_star * (
                    4 * s ** 4 * mod_z ** 4 - 8 * s ** 3 * mod_z ** 4 + 12 * s ** 3 * mod_z ** 2 + 5 * s ** 2 * mod_z ** 4 - 24 * s ** 2 * mod_z ** 2 + 6 * s ** 2 - s * mod_z ** 4 + 15 * s * mod_z ** 2 - 9 * s - 3 * mod_z ** 2 + 3) / (
                                 mod_z ** 10 + 5 * mod_z ** 8 + 10 * mod_z ** 6 + 10 * mod_z ** 4 + 5 * mod_z ** 2 + 1)
        moments[7] = s * (
                    -16 * s ** 4 * mod_z ** 6 + 16 * s ** 4 * mod_z ** 4 + 48 * s ** 3 * mod_z ** 6 - 80 * s ** 3 * mod_z ** 4 + 32 * s ** 3 * mod_z ** 2 - 36 * s ** 2 * mod_z ** 6 + 164 * s ** 2 * mod_z ** 4 - 72 * s ** 2 * mod_z ** 2 + 8 * s ** 2 + 8 * s * mod_z ** 6 - 112 * s * mod_z ** 4 + 76 * s * mod_z ** 2 - 4 * s + 24 * mod_z ** 4 - 24 * mod_z ** 2) / (
                                 mod_z ** 10 + 5 * mod_z ** 8 + 10 * mod_z ** 6 + 10 * mod_z ** 4 + 5 * mod_z ** 2 + 1)
        moments[8] = s * z * (
                    8 * s ** 4 * mod_z ** 6 - 16 * s ** 4 * mod_z ** 4 + 8 * s ** 4 * mod_z ** 2 - 20 * s ** 3 * mod_z ** 6 + 80 * s ** 3 * mod_z ** 4 - 52 * s ** 3 * mod_z ** 2 + 8 * s ** 3 + 16 * s ** 2 * mod_z ** 6 - 124 * s ** 2 * mod_z ** 4 + 120 * s ** 2 * mod_z ** 2 - 20 * s ** 2 - 4 * s * mod_z ** 6 + 76 * s * mod_z ** 4 - 104 * s * mod_z ** 2 + 16 * s - 16 * mod_z ** 4 + 28 * mod_z ** 2 - 4) / (
                                 1 * mod_z ** 10 + 5 * mod_z ** 8 + 10 * mod_z ** 6 + 10 * mod_z ** 4 + 5 * mod_z ** 2 + 1)
        moments[9] = s * z ** 2 * (
                    -4 * s ** 4 * mod_z ** 6 + 12 * s ** 4 * mod_z ** 4 - 12 * s ** 4 * mod_z ** 2 + 4 * s ** 4 + 2 * s ** 3 * mod_z ** 6 - 54 * s ** 3 * mod_z ** 4 + 78 * s ** 3 * mod_z ** 2 - 26 * s ** 3 + 56 * s ** 2 * mod_z ** 4 - 164 * s ** 2 * mod_z ** 2 + 60 * s ** 2 - 24 * s * mod_z ** 4 + 120 * s * mod_z ** 2 - 56 * s + 4 * mod_z ** 4 - 28 * mod_z ** 2 + 16) / (
                                 1 * mod_z ** 10 + 5 * mod_z ** 8 + 10 * mod_z ** 6 + 10 * mod_z ** 4 + 5 * mod_z ** 2 + 1)
        moments[10] = 8 * s * z_star ** 3 * (2 * s ** 2 * mod_z ** 2 - s * mod_z ** 2 + 4 * s - 2) * (
                    2 * s ** 2 - 5 * s + 3) / (
                                  mod_z ** 10 + 5 * mod_z ** 8 + 10 * mod_z ** 6 + 10 * mod_z ** 4 + 5 * mod_z ** 2 + 1)
        moments[11] = s * z_star ** 2 * (
                    -16 * s ** 4 * mod_z ** 4 + 16 * s ** 4 * mod_z ** 2 + 72 * s ** 3 * mod_z ** 4 - 64 * s ** 3 * mod_z ** 2 + 24 * s ** 3 - 80 * s ** 2 * mod_z ** 4 + 164 * s ** 2 * mod_z ** 2 - 36 * s ** 2 + 24 * s * mod_z ** 4 - 164 * s * mod_z ** 2 + 12 * s + 48 * mod_z ** 2) / (
                                  mod_z ** 10 + 5 * mod_z ** 8 + 10 * mod_z ** 6 + 10 * mod_z ** 4 + 5 * mod_z ** 2 + 1)
        moments[12] = s * z_star * (
                    8 * s ** 4 * mod_z ** 6 - 16 * s ** 4 * mod_z ** 4 + 8 * s ** 4 * mod_z ** 2 - 36 * s ** 3 * mod_z ** 6 + 80 * s ** 3 * mod_z ** 4 - 36 * s ** 3 * mod_z ** 2 + 8 * s ** 3 + 48 * s ** 2 * mod_z ** 6 - 140 * s ** 2 * mod_z ** 4 + 88 * s ** 2 * mod_z ** 2 - 4 * s ** 2 - 16 * s * mod_z ** 6 + 124 * s * mod_z ** 4 - 60 * s * mod_z ** 2 - 36 * mod_z ** 4 + 12 * mod_z ** 2) / (
                                  mod_z ** 10 + 5 * mod_z ** 8 + 10 * mod_z ** 6 + 10 * mod_z ** 4 + 5 * mod_z ** 2 + 1)
        moments[13] = s * (
                    -4 * s ** 4 * mod_z ** 8 + 12 * s ** 4 * mod_z ** 6 - 12 * s ** 4 * mod_z ** 4 + 4 * s ** 4 * mod_z ** 2 + 12 * s ** 3 * mod_z ** 8 - 62 * s ** 3 * mod_z ** 6 + 66 * s ** 3 * mod_z ** 4 - 18 * s ** 3 * mod_z ** 2 + 2 * s ** 3 - 12 * s ** 2 * mod_z ** 8 + 104 * s ** 2 * mod_z ** 6 - 128 * s ** 2 * mod_z ** 4 + 36 * s ** 2 * mod_z ** 2 + 4 * s * mod_z ** 8 - 68 * s * mod_z ** 6 + 108 * s * mod_z ** 4 - 20 * s * mod_z ** 2 + 16 * mod_z ** 6 - 28 * mod_z ** 4 + 4 * mod_z ** 2) / (
                                  1 * mod_z ** 10 + 5 * mod_z ** 8 + 10 * mod_z ** 6 + 10 * mod_z ** 4 + 5 * mod_z ** 2 + 1)
        moments[14] = s * z * (
                    2 * s ** 4 * mod_z ** 8 - 8 * s ** 4 * mod_z ** 6 + 12 * s ** 4 * mod_z ** 4 - 8 * s ** 4 * mod_z ** 2 + 2 * s ** 4 + 32 * s ** 3 * mod_z ** 6 - 72 * s ** 3 * mod_z ** 4 + 48 * s ** 3 * mod_z ** 2 - 8 * s ** 3 - 28 * s ** 2 * mod_z ** 6 + 140 * s ** 2 * mod_z ** 4 - 100 * s ** 2 * mod_z ** 2 + 12 * s ** 2 + 12 * s * mod_z ** 6 - 96 * s * mod_z ** 4 + 84 * s * mod_z ** 2 - 8 * s - 2 * mod_z ** 6 + 22 * mod_z ** 4 - 22 * mod_z ** 2 + 2) / (
                                  1 * mod_z ** 10 + 5 * mod_z ** 8 + 10 * mod_z ** 6 + 10 * mod_z ** 4 + 5 * mod_z ** 2 + 1)
        moments[15] = 8 * s * z_star ** 5 * (2 * s - 1) * (2 * s ** 3 - 9 * s ** 2 + 13 * s - 6) / (
                    mod_z ** 10 + 5 * mod_z ** 8 + 10 * mod_z ** 6 + 10 * mod_z ** 4 + 5 * mod_z ** 2 + 1)
        moments[16] = s * z_star ** 4 * (
                    -8 * s ** 2 * mod_z ** 2 + 8 * s ** 2 + 36 * s * mod_z ** 2 - 4 * s - 16 * mod_z ** 2) * (
                                  2 * s ** 2 - 5 * s + 3) / (
                                  mod_z ** 10 + 5 * mod_z ** 8 + 10 * mod_z ** 6 + 10 * mod_z ** 4 + 5 * mod_z ** 2 + 1)
        moments[17] = s * z_star ** 3 * (
                    8 * s ** 4 * mod_z ** 4 - 16 * s ** 4 * mod_z ** 2 + 8 * s ** 4 - 60 * s ** 3 * mod_z ** 4 + 88 * s ** 3 * mod_z ** 2 - 12 * s ** 3 + 148 * s ** 2 * mod_z ** 4 - 128 * s ** 2 * mod_z ** 2 + 4 * s ** 2 - 132 * s * mod_z ** 4 + 68 * s * mod_z ** 2 + 36 * mod_z ** 4 - 12 * mod_z ** 2) / (
                                  mod_z ** 10 + 5 * mod_z ** 8 + 10 * mod_z ** 6 + 10 * mod_z ** 4 + 5 * mod_z ** 2 + 1)
        moments[18] = s * z_star ** 2 * (
                    -4 * s ** 4 * mod_z ** 6 + 12 * s ** 4 * mod_z ** 4 - 12 * s ** 4 * mod_z ** 2 + 4 * s ** 4 + 26 * s ** 3 * mod_z ** 6 - 78 * s ** 3 * mod_z ** 4 + 54 * s ** 3 * mod_z ** 2 - 2 * s ** 3 - 60 * s ** 2 * mod_z ** 6 + 164 * s ** 2 * mod_z ** 4 - 56 * s ** 2 * mod_z ** 2 + 56 * s * mod_z ** 6 - 120 * s * mod_z ** 4 + 24 * s * mod_z ** 2 - 16 * mod_z ** 6 + 28 * mod_z ** 4 - 4 * mod_z ** 2) / (
                                  mod_z ** 10 + 5 * mod_z ** 8 + 10 * mod_z ** 6 + 10 * mod_z ** 4 + 5 * mod_z ** 2 + 1)
        moments[19] = s * z_star * (
                    2 * s ** 4 * mod_z ** 8 - 8 * s ** 4 * mod_z ** 6 + 12 * s ** 4 * mod_z ** 4 - 8 * s ** 4 * mod_z ** 2 + 2 * s ** 4 - 8 * s ** 3 * mod_z ** 8 + 48 * s ** 3 * mod_z ** 6 - 72 * s ** 3 * mod_z ** 4 + 32 * s ** 3 * mod_z ** 2 + 12 * s ** 2 * mod_z ** 8 - 100 * s ** 2 * mod_z ** 6 + 140 * s ** 2 * mod_z ** 4 - 28 * s ** 2 * mod_z ** 2 - 8 * s * mod_z ** 8 + 84 * s * mod_z ** 6 - 96 * s * mod_z ** 4 + 12 * s * mod_z ** 2 + 2 * mod_z ** 8 - 22 * mod_z ** 6 + 22 * mod_z ** 4 - 2 * mod_z ** 2) / (
                                  1 * mod_z ** 10 + 5 * mod_z ** 8 + 10 * mod_z ** 6 + 10 * mod_z ** 4 + 5 * mod_z ** 2 + 1)
        moments[20] = s * (
                    -1 * s ** 4 * mod_z ** 10 + 5 * s ** 4 * mod_z ** 8 - 10 * s ** 4 * mod_z ** 6 + 10 * s ** 4 * mod_z ** 4 - 5 * s ** 4 * mod_z ** 2 + 1 * s ** 4 - 20 * s ** 3 * mod_z ** 8 + 60 * s ** 3 * mod_z ** 6 - 60 * s ** 3 * mod_z ** 4 + 20 * s ** 3 * mod_z ** 2 + 20 * s ** 2 * mod_z ** 8 - 120 * s ** 2 * mod_z ** 6 + 120 * s ** 2 * mod_z ** 4 - 20 * s ** 2 * mod_z ** 2 - 10 * s * mod_z ** 8 + 90 * s * mod_z ** 6 - 90 * s * mod_z ** 4 + 10 * s * mod_z ** 2 + 2 * mod_z ** 8 - 22 * mod_z ** 6 + 22 * mod_z ** 4 - 2 * mod_z ** 2) / (
                                  1 * mod_z ** 10 + 5 * mod_z ** 8 + 10 * mod_z ** 6 + 10 * mod_z ** 4 + 5 * mod_z ** 2 + 1)
    elif order == 4:
        moments = MutableDenseNDimArray(np.zeros(15), (15,))
        moments[0] = 4 * s * z ** 4 * (2 * s - 1) * (2 * s ** 2 - 5 * s + 3) / (
                    mod_z ** 8 + 4 * mod_z ** 6 + 6 * mod_z ** 4 + 4 * mod_z ** 2 + 1)
        moments[1] = 4 * s * z ** 2 * (
                    4 * s ** 3 * mod_z ** 2 - 6 * s ** 2 * mod_z ** 2 + 6 * s ** 2 + 2 * s * mod_z ** 2 - 9 * s + 3) / (
                                 mod_z ** 8 + 4 * mod_z ** 6 + 6 * mod_z ** 4 + 4 * mod_z ** 2 + 1)
        moments[2] = s * z ** 3 * (
                    -8 * s ** 3 * mod_z ** 2 + 8 * s ** 3 + 12 * s ** 2 * mod_z ** 2 - 36 * s ** 2 - 4 * s * mod_z ** 2 + 40 * s - 12) / (
                                 mod_z ** 8 + 4 * mod_z ** 6 + 6 * mod_z ** 4 + 4 * mod_z ** 2 + 1)
        moments[3] = 4 * s * (
                    4 * s ** 3 * mod_z ** 4 - 4 * s ** 2 * mod_z ** 4 + 8 * s ** 2 * mod_z ** 2 + s * mod_z ** 4 - 8 * s * mod_z ** 2 + 2 * s + 2 * mod_z ** 2 - 1) / (
                                 mod_z ** 8 + 4 * mod_z ** 6 + 6 * mod_z ** 4 + 4 * mod_z ** 2 + 1)
        moments[4] = s * z * (
                    -8 * s ** 3 * mod_z ** 4 + 8 * s ** 3 * mod_z ** 2 + 12 * s ** 2 * mod_z ** 4 - 28 * s ** 2 * mod_z ** 2 + 8 * s ** 2 - 4 * s * mod_z ** 4 + 28 * s * mod_z ** 2 - 12 * s - 8 * mod_z ** 2 + 4) / (
                                 mod_z ** 8 + 4 * mod_z ** 6 + 6 * mod_z ** 4 + 4 * mod_z ** 2 + 1)
        moments[5] = s * z ** 2 * (
                    4 * s ** 3 * mod_z ** 4 - 8 * s ** 3 * mod_z ** 2 + 4 * s ** 3 - 2 * s ** 2 * mod_z ** 4 + 28 * s ** 2 * mod_z ** 2 - 18 * s ** 2 - 20 * s * mod_z ** 2 + 24 * s + 4 * mod_z ** 2 - 8) / (
                                 mod_z ** 8 + 4 * mod_z ** 6 + 6 * mod_z ** 4 + 4 * mod_z ** 2 + 1)
        moments[6] = 4 * s * z_star ** 2 * (
                    4 * s ** 3 * mod_z ** 2 - 6 * s ** 2 * mod_z ** 2 + 6 * s ** 2 + 2 * s * mod_z ** 2 - 9 * s + 3) / (
                                 mod_z ** 8 + 4 * mod_z ** 6 + 6 * mod_z ** 4 + 4 * mod_z ** 2 + 1)
        moments[7] = s * z_star * (
                    -8 * s ** 3 * mod_z ** 4 + 8 * s ** 3 * mod_z ** 2 + 20 * s ** 2 * mod_z ** 4 - 20 * s ** 2 * mod_z ** 2 + 8 * s ** 2 - 8 * s * mod_z ** 4 + 32 * s * mod_z ** 2 - 4 * s - 12 * mod_z ** 2) / (
                                 mod_z ** 8 + 4 * mod_z ** 6 + 6 * mod_z ** 4 + 4 * mod_z ** 2 + 1)
        moments[8] = s * (
                    4 * s ** 3 * mod_z ** 6 - 8 * s ** 3 * mod_z ** 4 + 4 * s ** 3 * mod_z ** 2 - 8 * s ** 2 * mod_z ** 6 + 26 * s ** 2 * mod_z ** 4 - 12 * s ** 2 * mod_z ** 2 + 2 * s ** 2 + 4 * s * mod_z ** 6 - 24 * s * mod_z ** 4 + 16 * s * mod_z ** 2 + 8 * mod_z ** 4 - 4 * mod_z ** 2) / (
                                 1 * mod_z ** 8 + 4 * mod_z ** 6 + 6 * mod_z ** 4 + 4 * mod_z ** 2 + 1)
        moments[9] = s * z * (
                    -2 * s ** 3 * mod_z ** 6 + 6 * s ** 3 * mod_z ** 4 - 6 * s ** 3 * mod_z ** 2 + 2 * s ** 3 - 18 * s ** 2 * mod_z ** 4 + 24 * s ** 2 * mod_z ** 2 - 6 * s ** 2 + 10 * s * mod_z ** 4 - 28 * s * mod_z ** 2 + 6 * s - 2 * mod_z ** 4 + 8 * mod_z ** 2 - 2) / (
                                 1 * mod_z ** 8 + 4 * mod_z ** 6 + 6 * mod_z ** 4 + 4 * mod_z ** 2 + 1)
        moments[10] = 4 * s * z_star ** 4 * (2 * s - 1) * (2 * s ** 2 - 5 * s + 3) / (
                    mod_z ** 8 + 4 * mod_z ** 6 + 6 * mod_z ** 4 + 4 * mod_z ** 2 + 1)
        moments[11] = s * z_star ** 3 * (
                    -8 * s ** 3 * mod_z ** 2 + 8 * s ** 3 + 36 * s ** 2 * mod_z ** 2 - 12 * s ** 2 - 40 * s * mod_z ** 2 + 4 * s + 12 * mod_z ** 2) / (
                                  mod_z ** 8 + 4 * mod_z ** 6 + 6 * mod_z ** 4 + 4 * mod_z ** 2 + 1)
        moments[12] = s * z_star ** 2 * (
                    4 * s ** 3 * mod_z ** 4 - 8 * s ** 3 * mod_z ** 2 + 4 * s ** 3 - 18 * s ** 2 * mod_z ** 4 + 28 * s ** 2 * mod_z ** 2 - 2 * s ** 2 + 24 * s * mod_z ** 4 - 20 * s * mod_z ** 2 - 8 * mod_z ** 4 + 4 * mod_z ** 2) / (
                                  mod_z ** 8 + 4 * mod_z ** 6 + 6 * mod_z ** 4 + 4 * mod_z ** 2 + 1)
        moments[13] = s * z_star * (
                    -2 * s ** 3 * mod_z ** 6 + 6 * s ** 3 * mod_z ** 4 - 6 * s ** 3 * mod_z ** 2 + 2 * s ** 3 + 6 * s ** 2 * mod_z ** 6 - 24 * s ** 2 * mod_z ** 4 + 18 * s ** 2 * mod_z ** 2 - 6 * s * mod_z ** 6 + 28 * s * mod_z ** 4 - 10 * s * mod_z ** 2 + 2 * mod_z ** 6 - 8 * mod_z ** 4 + 2 * mod_z ** 2) / (
                                  mod_z ** 8 + 4 * mod_z ** 6 + 6 * mod_z ** 4 + 4 * mod_z ** 2 + 1)
        moments[14] = s * (
                    1 * s ** 3 * mod_z ** 8 - 4 * s ** 3 * mod_z ** 6 + 6 * s ** 3 * mod_z ** 4 - 4 * s ** 3 * mod_z ** 2 + 1 * s ** 3 + 12 * s ** 2 * mod_z ** 6 - 24 * s ** 2 * mod_z ** 4 + 12 * s ** 2 * mod_z ** 2 - 8 * s * mod_z ** 6 + 28 * s * mod_z ** 4 - 8 * s * mod_z ** 2 + 2 * mod_z ** 6 - 8 * mod_z ** 4 + 2 * mod_z ** 2) / (
                                  1 * mod_z ** 8 + 4 * mod_z ** 6 + 6 * mod_z ** 4 + 4 * mod_z ** 2 + 1)
    elif order == 3:
        moments = MutableDenseNDimArray(np.zeros(10), (10,))
        moments[0] = 2*s * (2*s - 1) * (2 * s - 2) * z ** 3 / (1 + mod_z ** 2) ** 3
        moments[1] = 4*s * (2*s - 1) * z * (s * mod_z ** 2 + 1) / (1 + mod_z ** 2) ** 3
        moments[2] = -2*s * (2*s - 1) * z ** 2 * (s * mod_z ** 2 - s + 2) / (1 + mod_z ** 2) ** 3
        moments[3] = 4*s * (2*s - 1) * z_star * (s * mod_z ** 2 + 1) / (1 + mod_z ** 2) ** 3
        moments[4] = 2*s * (-2*s * (s - 1) * mod_z ** 4 + (s * (2*s - 3) + 2) * mod_z ** 2 + s) / (
                    1 + mod_z ** 2) ** 3
        moments[5] = 2*s * z * (s ** 2 * mod_z ** 4 + (-2*s * (s - 2) - 1) * mod_z ** 2 + (s - 1) ** 2) / (
                    1 + mod_z ** 2) ** 3
        moments[6] = 2*s * (2*s - 1) * (2 * s - 2) * z_star ** 3 / (1 + mod_z ** 2) ** 3
        moments[7] = -2*s * (2*s - 1) * z_star ** 2 * ((s - 2) * mod_z ** 2 - s) / (1 + mod_z ** 2) ** 3
        moments[8] = 2*s * z_star * ((s - 1) ** 2 * mod_z ** 4 + (-2*s * (s - 2) - 1) * mod_z ** 2 + s ** 2) / (
                    1 + mod_z ** 2) ** 3
        moments[9] = -s * (mod_z ** 2 - 1) * (s ** 2 * (mod_z ** 4 + 1) - 2 * (s * (s - 3) + 1) * mod_z ** 2) / (
                    1 + mod_z ** 2) ** 3
    elif order == 2:
        moments = MutableDenseNDimArray(np.zeros(6), (6,))
        moments[0] = 2*s * (2*s - 1) * z ** 2 / (1 + mod_z ** 2) ** 2
        moments[1] = 2*s * (2*s * mod_z ** 2 + 1) / (1 + mod_z ** 2) ** 2
        moments[2] = 2*s * z * (-1 + s - s * mod_z ** 2) / (1 + mod_z ** 2) ** 2
        moments[3] = 2*s * (2*s - 1) * z_star ** 2 / (1 + mod_z ** 2) ** 2
        moments[4] = 2*s * z_star * (s + (1 - s) * mod_z ** 2) / (1 + mod_z ** 2) ** 2
        moments[5] = (s ** 2 * (1 - mod_z ** 2) ** 2 + 2 * s * mod_z ** 2) / (1 + mod_z ** 2) ** 2
    elif order == 1:
        moments = MutableDenseNDimArray(np.zeros(3), (3,))
        moments[0] = 2*s * z / (1 + mod_z ** 2)
        moments[1] = 2*s * z_star / (1 + mod_z ** 2)
        moments[2] = s * (1 - mod_z ** 2) / (1 + mod_z ** 2)
    else:
        print("order not supported" + str(order))
    return moments


def rearrange_with_commutators_any_order(function, s_up, s_down, s_z, order):
    """Returns the properly ordered expression of function to be replaced using moments in compute_moments_list_complex"""
    site_separated = function
    for i in range(binomial(order + 1, 2)):
        first_step = expand(site_separated).subs(s_down * s_up, s_up * s_down - 2 * s_z).subs(s_z * s_up,
                                                            s_up * s_z + s_up).subs(s_z * s_down, s_down * s_z - s_down)
        site_separated = first_step
    return simplify(site_separated)


def replace_moments_any_order(func, order, moments, s_up, s_down, s_z):
    """Returs expression where moments are replaced with spin coherent state unit vector ones, from properly ordered expression"""
    size = binomial(order + 2, 2)
    spin_op_list = MutableDenseNDimArray(np.zeros(size), (size,))
    N = order
    arrays = generate_and_sort_arrays_with_sum(N)
    spin_elements = np.array(([s_up, s_down, s_z]))
    i = 0
    moments_replaced = func
    for array in arrays:
        result = 1
        for j in range(3):
            result = result * spin_elements[j] ** array[j]
        spin_op_list[i] = expand(rearrange_with_commutators_any_order(result, s_up, s_down, s_z, order))
        i = i+1
    for j in range(size):
        final_result = expand(moments_replaced).subs(spin_op_list[j], moments[j])
        moments_replaced = final_result
    return simplify(moments_replaced)


def as_function_of_complex(S, s, func, order, z, z_star, mod_z, s_up, s_down, s_z, L, J, K, moments_all):
    """Returns func with computed moments as function of the complex spin coherent states parameters"""
    final_result = func
    for w in reversed(range(1, order + 1)):
        # moments = compute_moments_list_complex_any_order(S, s, L, z, J, K, mod_z, z_star, w)
        moments = compute_moments_list_complex(S, s, L, z, J, K, mod_z, z_star, w)
        # moments = moments_all[w-1]
        final_result = replace_moments_any_order(final_result, w, moments, s_up, s_down, s_z)
    return final_result


def as_function_of_real(func, z, z_star, mod_z, n_x, n_y, n_z):
    """Returns func with computed moments as function of the real spin coherent state unit vector components parameters"""
    result = simplify(
        expand(func).subs(z, ((n_x + I * n_y) / 2) * (1 + mod_z ** 2)).subs(z_star,
                                ((n_x - I * n_y) / 2) * (1 + mod_z ** 2)).subs(mod_z * mod_z, (1 - n_z) / (1 + n_z)))
    return result


def generate_and_sort_arrays_with_sum(N):
    """Returns an array of all sorted and properly ordered power combinations for x_1^a*x_2^b*x_3^c with a+b+c = N"""
    result = []
    for a in range(N + 1):
        for b in range(N + 1):
            c = N - a - b
            if 0 <= c <= N:
                result.append([a, b, c])
    result.sort(key=lambda x: (-x[0], -x[1], -x[2]))

    return result


def compute_moments_list_cartesian_any_order(order, s_x, s_y, s_z, s_up, s_down):
    """Returns a list of the computed moments (subset of only properly ordered ones) in terms of
    the spin coherent state unit vector components for a given order
    """
    size = binomial(order + 2, 2)
    moments = MutableDenseNDimArray(np.zeros(size), (size,))
    N = order
    arrays = generate_and_sort_arrays_with_sum(N)
    spin_elements = np.array(([s_x, s_y, s_z]))
    elem = 0
    for array in arrays:
        result = 1
        for depth in range(3):
            result = result * spin_elements[depth] ** array[depth]
        moments[elem] = expand(rearrange_with_commutators_any_order(result, s_up, s_down, s_z, order))
        elem = elem + 1
    return moments


def compute_moments_real(S, s, order, s_x, s_y, s_z, s_up, s_down, z, z_star, mod_z, n_x, n_y, n_z, L, J, K, moments_all):
    """Returns a list of the computed moments (subset of only properly ordered ones) in terms of
    the spin coherent state unit vector components up to the given order using compute_moments_list_cartesian_any_order
    """
    moments = compute_moments_list_cartesian_any_order(order, s_x, s_y, s_z, s_up, s_down)
    for elem in range(len(moments)):
        moments[elem] = as_function_of_complex(S, s, moments[elem], order, z, z_star, mod_z, s_up, s_down, s_z, L, J, K, moments_all)
        moments[elem] = as_function_of_real(moments[elem], z, z_star, mod_z, n_x, n_y, n_z)
    return moments

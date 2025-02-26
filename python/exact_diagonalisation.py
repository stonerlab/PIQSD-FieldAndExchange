import numpy as np
from sympy import exp as exp
from sympy import Matrix, symbols, sqrt, eye, zeros, Rational, re
from sympy.physics.quantum import TensorProduct
from sympy.utilities import lambdify
from sympy.physics.quantum.dagger import Dagger
import matplotlib.pyplot as plt
import functools as ft
import scipy.constants as scp

electron_g_factor = -1.0 * scp.value("electron g factor")
muB = scp.value("Bohr magneton")
constant = electron_g_factor*muB/scp.k


def exact_thermal_spin_half_s_z(j, b_z, t):
    exchange_factor = j/scp.k
    return (1.0/2.0)*(np.exp(constant*b_z / t) - np.exp(-constant*b_z / t)) / (1 + np.exp(-constant*b_z / t) + np.exp(-exchange_factor / t) + np.exp(constant*b_z / t))


def kronecker_delta(a, b):
    """The Kronecker product. Returns 1 if A = B or returns 0 if A != B.
    Inputs:
        a (Integer or float): An integer or float value
        b (Integer or float): Another integer or float value
    Returns:
        1 or 0 if a=b or if a!=b respectively
    """
    if a == b:
        return 1
    else:
        return 0


def get_vector_eigenvalue(matrix, eigen_v):
    """This function returns the eigenvalue of a given n x 1 eigenvector v with respect to the given n x n matrix a.
    This function assumes that v is an eigenstate of a, it should not be
    used to determine if v is an eigenstate of a.
    Inputs:
        a (Sympy matrix): Square n x n matrix.
        v (Sympy matrix): n x 1 matrix representing the eigenvector which you want to find the eigenvalue for.
    Outputs:
        Eigenvalue (float): The eigenvalue of v with respect to a
    """
    state = Matrix.multiply(matrix, eigen_v)
    count = 0

    for i in range(0, len(eigen_v)):
        if state[i] == 0:
            count = count + 1
        else:
            index = i
            break

    if count == len(eigen_v):
        return 0
    else:
        eigenvalue = state[index] / eigen_v[index]

    for i in range(0, len(eigen_v)):
        if state[i] != eigenvalue * eigen_v[i]:
            raise Exception(f"v is not an Eigenstate of a, at i={i}, a_i/v_i={matrix[i] / eigen_v[i]}! If you are getting this error then it is likely that one of the degenerate eigenvectors which SymPy has returned is not an eigenstate of the total Sz operator. This error was raised to avoid returning incorrect Sz eigenvalues.Try using the numerical model.")
    else:
        return eigenvalue


def pauli_x_elements(j, l, s):
    """Finds the j,l elements of the Pauli-x matrix for a particle of spin s/2 using:
    TAH, RAJDEEP. 2020. “Calculating the Pauli Matrix Equivalent for Spin-1 Particles and Further Implementing It to Calculate the Unitary Operators of the Harmonic Oscillator Involving a Spin-1 System.” IndiaRxiv. August 5. doi:10.35543/osf.io/6ck3w
    j and l range from +s/2 to -s/2 in increments of 1, i.e. the first element
    of the Pauli-x matrix, usually denoted Sx_j,l=Sx_1,1 is instead denoted Sx_j,l=Sx_(s/2,s/2). For example, for the s=2 case (spin-1), the first element would be denoted Sx_(1,1), the
    second Sx_(1,0) the third Sx_(1,-1) and so on.
    Inputs:
        j (Integer / float): One of the multiple z-component spins which can be measured for a single spin for a given s (spin-s/2), used to find the matrix element of the Pauli-x matrix
        l (Integer / float): One of the multiple z-component spins which can be measured for a single spin for a given s (spin-s/2), used to find the matrix element of the Pauli-x matrix
        s (Integer): The principle quantum spin number s for a spin-s/2 particle
    Outputs:
        Matrix element Sx_(j,l) (Float): The j,l matrix element of the Pauli-x matrix for the principal quantum number s.
    """
    spin = Rational(s, 2)

    s = spin * (spin + 1)
    j_neg = j * (j - 1)
    j_pos = j * (j + 1)
    part_a = (sqrt(s - j_neg) / (2 * spin)) * kronecker_delta(j, l + 1)
    part_b = (sqrt(s - j_pos) / (2 * spin)) * kronecker_delta(j, l - 1)
    return part_a + part_b


def pauli_y_elements(j, l, s):
    """Finds the j,l elements of the Pauli-y matrix for a particle of spin s/2 using:
    TAH, RAJDEEP. 2020. “Calculating the Pauli Matrix Equivalent for Spin-1 Particles and Further Implementing It to Calculate the Unitary Operators of the Harmonic Oscillator Involving a Spin-1 System.” IndiaRxiv. August 5. doi:10.35543/osf.io/6ck3w
    j and l range from +s/2 to -s/2 in increments of 1, i.e. the first element
    of the Pauli-y matrix, usually denoted Sy_j,l=Sy_1,1 is instead denoted Sy_j,l=Sy_(s/2,s/2). For example, for the s=2 case (spin-1), the first element would be denoted Sy_(1,1), the
    second Sy_(1,0), the third Sy_(1,-1) and so on.
    Inputs:
        j (Integer / float): One of the multiple z-component spins which can be measured for a single spin for a given s (spin-s/2), used to find the matrix element of the Pauli-y matrix
        l (Integer / float): One of the multiple z-component spins which can be measured for a single spin for a given s (spin-s/2), used to find the matrix element of the Pauli-y matrix
        s (Integer): The principal quantum spin number s for a spin-s/2 particle
    Outputs:
        Matrix element sy(j,l) (Float): The j,l matrix element of the Pauli-y matrix for the principal quantum number s.
    """
    spin = Rational(s, 2)

    s = spin * (spin + 1)
    j_neg = j * (j - 1)
    j_pos = j * (j + 1)
    part_a = (sqrt(s - j_neg) / (2j * spin)) * kronecker_delta(j, l + 1)
    part_b = (sqrt(s - j_pos) / (2j * spin)) * kronecker_delta(j, l - 1)
    return part_a - part_b


def get_pauli_matrices(s):
    """Calculates and returns the basic x,y,z spin operator matrices. This is done by calculating the possible z-component spin values (l, also referred to as j) for a given s and looping
    through them, each time calling the function sx_element and sy_element to calculate the value for the Pauli matrices element for a given j,l. A list of these elements is then used to determine build
    the Pauli matrix using SymPy. In the case for the Spin-z operator, the matrix is just the diagonal matrix with diagonal values equal to the values of j.
    Inputs:
        s (Integer / float): The principle quantum spin number s for a spin-s/2 particle
    Outputs:
        sx, sy, sz (Sympy matrices): Returns the spin-x, spin-y and spin-z for the given s (acting on one spin) operators respectively.
    """
    sx_elements = list()
    sy_elements = list()
    sz_elements = list()

    spin = Rational(s, 2)
    # spin=s/2
    j = int(s + 1)

    for i in range(0, j):
        sz_elements.append(spin - i)
    sz = Matrix.diag(sz_elements)

    for i in range(0, j):
        for k in range(0, j):
            sx_elements.append(pauli_x_elements(sz_elements[i], sz_elements[k], s))
            sy_elements.append(pauli_y_elements(sz_elements[i], sz_elements[k], s))
    sx = spin * Matrix(s + 1, s + 1, sx_elements)
    sy = spin * Matrix(s + 1, s + 1, sy_elements)

    return sx, sy, sz


def find_s_acting_on_i(i, spins_number, s_matrix):
    """Find a given spin operator S (i.e. Sx,Sy or Sz) acting on the ith spin in the chain of N spins. This will be used in the Hamiltonian and to find the total Sz operator or total spin operator squared
    Inputs:
        i (Integer): An integer representing the position of the ith spin which the operator is acting on (the first spin is spin is indexed 1, the last N).
        spins_number (Integer): An integer representing the number of spin in the chain.
        s_matrix (Sympy matrix): A matrix of a basic spin operator for a system on principal quantum spin number s.
    Outputs:
        s_i (Sympy matrix): A matrix representing the given basic spin operator acting on the ith spin in the chain
    """
    dimensions_of_s = int(sqrt(len(s_matrix)))
    identity_matrix = eye(dimensions_of_s)
    if spins_number <= 0:
        raise Exception("N must be a real positive integer to represent a real physical system of N spins.")
    if i <= 0:
        raise Exception("i must be 0 < i <= spins_number. You cannot calculate the interaction of the zeroth spin in a chain.")
    if i > spins_number:
        raise Exception("i cannot be bigger than spins_number. The position of the spin must be confined in the chain.")

    tensor_list = list()
    for j in range(1, (i-1) + 1):
        tensor_list.append(identity_matrix)
    tensor_list.append(s_matrix)
    for j in range(1, (spins_number - i) + 1):
        tensor_list.append(identity_matrix)

    s_i = ft.reduce(TensorProduct, tensor_list)
    return s_i


def si_sj_acting_on_ij(i, j, spins_number, sx, sy, sz):
    """Finds the dot product between two spin vector operators each acting on the ith and jth spin respectively. i.e. Finds S^(i)dot S^(j). This is done for a given i, j and spins_number.
    This can be used to find S^(i) dot S^(i+1). j should ideally be > i. But if i>j, then j and i will be swapped and S^(j) dot S^(i)
    will be returned. (spin operators for different sites commute).
    Inputs:
        i (Integer): An integer representing the position of the ith spin which the spin vector operator S^(i) is acting on (the first spin is indexed 1, the last N). i must be 0<i<=N
        j (Integer): An integer representing the position of the jth spin which the spin vector operator S^(j) is acting on. j must be i<j<=N
        spins_number (Integer): The number of spins in the chain, loop or cubic lattice.
        sx (SymPy Matrix): A SymPy matrix representing the basic x-spin operator for a given s
        sy (SymPy Matrix): A SymPy matrix representing the basic y-spin operator for a given s
        sz (SymPy Matrix): A SymPy matrix representing the basic z-spin operator for a given s
    Outputs:
        si_sj (SymPy Matrix): A SymPy matrix representing the dot product between the spin vector acting on the ith spin and the spin operator acting on the jth particle.
        """
    if i > spins_number or j > spins_number:
        raise Exception("i and j should not be larger than N")
    elif i > j:
        temp_j = j
        j = i
        i = temp_j
    elif i <= 0 or j <= 0 or spins_number <= 0:
        raise Exception("i,j and N should not be equal to or less than zero!")
    elif i == j:
        raise Exception("i should not be equal to j!")

    sx_dimensions = int(sqrt(len(sx)))
    identity_matrix = eye(sx_dimensions)
    sx_identity_sx_list = [sx]
    sy_identity_sy_list = [sy]
    sz_identity_sz_list = [sz]
    front_identity_list = list()
    end_identity_list = list()

    for p in range(1, (i-1)+1):
        front_identity_list.append(identity_matrix)

    for p in range(1, (j-i-1)+1):
        sx_identity_sx_list.append(identity_matrix)
        sy_identity_sy_list.append(identity_matrix)
        sz_identity_sz_list.append(identity_matrix)
    sx_identity_sx_list.append(sx)
    sy_identity_sy_list.append(sy)
    sz_identity_sz_list.append(sz)

    for p in range(1, (spins_number - j) + 1):
        end_identity_list.append(identity_matrix)

    sx_identity_sx = ft.reduce(TensorProduct, sx_identity_sx_list)
    sy_identity_sy = ft.reduce(TensorProduct, sy_identity_sy_list)
    sz_identity_sz = ft.reduce(TensorProduct, sz_identity_sz_list)

    sx_identity_sx_sy_identity_sy = sx_identity_sx + sy_identity_sy

    for p in range(0, len(sx_identity_sx_sy_identity_sy)):
        element = sqrt((re(sx_identity_sx_sy_identity_sy[p]))**2)
        if element < 10**(-14):
            sx_identity_sx_sy_identity_sy[p] = 0
    middle_tensor = sx_identity_sx_sy_identity_sy + sz_identity_sz

    final_tensor_list = list()

    for p in range(0, len(front_identity_list)):
        final_tensor_list.append(front_identity_list[p])
    final_tensor_list.append(middle_tensor)
    for q in range(0, len(end_identity_list)):
        final_tensor_list.append(end_identity_list[q])

    si_sj = ft.reduce(TensorProduct, final_tensor_list)
    return si_sj


def diagonalise(hamiltonian):
    """This function finds the eigenvalues and eigenvectors of a given Hamiltonian, then uses the eigenvalues to form a diagonal matrix in the basis of the eigenvalues. This function
    returns the eigenvalues, normalised eigenvectors and diagonalised Hamiltonian respectively.
    Inputs:
        hamiltonian (Sympy matrix): A square matrix representing the Hamiltonian of the system.
    Outputs:
        eigenvalues (List): A list of the eigenvalues of the Hamiltonian.
        n_eigenvectors (List of Sympy matrices): A list containing matrices which represent the eigenvectors of the Hamiltonian. The indexing of the list corresponds to the indexing
                                                        of the eigenvalues (i.e. the 2nd eigenvalue in the list of eigenvalues corresponds to the 2nd eigenvector in the list of eigenvectors)
        d_hamiltonian (Sympy matrix): A Sympy matrix representing the diagonalised Hamiltonian.
    """
    eigenvalues = []
    eigenvectors = []
    eigen = hamiltonian.eigenvects(chop=True)
    multiplicity_list = list()
    for i in range(0, len(eigen)):
        multiplicity = eigen[i][1]
        multiplicity_list.append(multiplicity)
        if multiplicity == 1:
            eigenvalues.append(eigen[i][0])
            eigenvectors.append(eigen[i][2][0])
        else:
            for j in range(0, multiplicity):
                eigenvalues.append(eigen[i][0])
                eigenvectors.append(eigen[i][2][j])

    eigenvectors_simplified = list()
    n_eigenvectors = []
    for i in range(0, len(eigenvectors)):
        eigenvector = eigenvectors[i]
        for p in range(0, len(eigenvector)):
            eigenvector[p] = eigenvector[p].expand()
        eigenvectors_simplified.append(eigenvector)
        magnitude = sqrt(Matrix.multiply(Dagger(eigenvector), eigenvector)[0])
        n_eigenvectors.append((1/magnitude)*eigenvector)

    d_hamiltonian = Matrix.diag(eigenvalues)
    return eigenvalues, n_eigenvectors, d_hamiltonian


def find_magnetisation(s, l):
    """A function which finds the total z-component spin expectation value of the composite spin system for spins with spin quantum number s (i.e. for a two spin system of spin-s/2). This is done by
    finding the Hamiltonian of the system, obtaining its eigenvectors and respective eigenvalues, then diagonalising the Hamiltonian using the diagonalise() function. The eigenvalues are then exponentiated
    and used to find the partition function and z-component spin expectation value (using their spectral decomposition equations, the exponentiated Hamiltonian is not explicitly used). The total
    z-component spin expectation value is normalised (with spin and the number of spins) so that it has a maximum value of 1. This function can find the Hamiltonian of a chain
    with nearest neighbours interactions. This exact model can find the magnetisation of systems of chains of two spins of up to around s=9 (depending on computational ressources), chains of spins (more than two spins) with s=1 (spin 1/2).
    Inputs:
        s (Integer): The principle spin quantum number s (s for spin/2)
        l (Integer): This is the number of spins in the system.
    Outputs:
        magnetisation (Function): A function of the normalised total z-component spin expectation value of the composite system spin-s/2 spins in terms of j (spin exchange interaction strength),g
                                  (Lande g-factor) , b (magnitude of magnetic field in z direction) and t (temperature). This function can be used to obtain numerical values for the magnetisation
                                    for given values of j, b and t.
        magnetisation_sympy (Sympy expression): A SymPy expression for the normalised total z-component spin expectation value of the composite spin system in terms of j, b, t. This is returned
                                                to compare this functions results with previous results."""

    b = symbols('B_z', real=True)
    t = symbols('T', real=True)
    j = symbols('J', real=True)  # in units of g * mu_b

    k_b = scp.k
    mu_b = scp.value("Bohr magneton")
    g = electron_g_factor

    hamiltonian_dimensions = int((s+1)**l)

    sx, sy, sz = get_pauli_matrices(s)

    sz_total = zeros(hamiltonian_dimensions)
    for i in range(1, l+1):
        sz_total = sz_total + find_s_acting_on_i(i, l, sz)

    sum_of_si_dot_si1 = zeros(hamiltonian_dimensions)
    for p in range(1, l):
        sum_of_si_dot_si1 = sum_of_si_dot_si1 + si_sj_acting_on_ij(p, p + 1, l, sx, sy, sz)

    hamiltonian = -g * mu_b * b * sz_total - j * g * mu_b * sum_of_si_dot_si1

    eigenvalues, n_eigenvectors, d_hamiltonian = diagonalise(hamiltonian)

    sz_sum = 0

    for p in range(0, len(eigenvalues)):
        total_z_spin = get_vector_eigenvalue(sz_total, n_eigenvectors[p])
        hamiltonian_element = exp(-eigenvalues[p] / (k_b * t))
        sz_sum = sz_sum + (total_z_spin * hamiltonian_element)

    z = 0
    for p in range(0, len(eigenvalues)):
        exponent = (-1.0 * eigenvalues[p]) / (k_b * t)
        exponential_eigenvalue = exp(exponent)
        z = z + exponential_eigenvalue
    magnetisation_sympy = Rational(1, l) * (sz_sum/z)

    magnetisation = lambdify((j, b, t), magnetisation_sympy)

    return magnetisation


def main():
    spin = 1  # Multiple of half integer, i.e. spin = 1 for 1/2.
    spin_chain_length = 2
    j_exchange = 1.0  # In units of g 8 mu_b
    field_z = 1.0  # In Tesla
    magnetisation = find_magnetisation(spin, spin_chain_length)

    temperature_array = np.linspace(0.1, 30, 2000)  # In Kelvin

    exact_magnetisation = magnetisation(j_exchange, field_z, temperature_array)

    plt.title(r"Total z-Component Spin Expectation Value", size=15)
    plt.plot(temperature_array, exact_magnetisation, "r-")
    plt.ylabel(r"$\langle \hat{S}^{T}_z \rangle /s$ ($\hbar$)", size=16)
    plt.xlabel(r"Temperature, $T$ (K)", size=16)
    plt.show()


if __name__ == "__main__":
    main()

import numpy as np
import cupy as cp
import scipy.special as sp

# For debug
# import matplotlib.pyplot as plt

# Legendre-Gauss-Lobatto nodes and quadrature weights dictionaries
lgl_nodes = {
    1: [0],
    2: [-1, 1],
    3: [-1, 0, 1],
    4: [-1, -np.sqrt(1 / 5), np.sqrt(1 / 5), 1],
    5: [-1, -np.sqrt(3 / 7), 0, np.sqrt(3 / 7), 1],
    6: [-1, -np.sqrt(1 / 3 + 2 * np.sqrt(7) / 21), -np.sqrt(1 / 3 - 2 * np.sqrt(7) / 21),
        np.sqrt(1 / 3 - 2 * np.sqrt(7) / 21), np.sqrt(1 / 3 + 2 * np.sqrt(7) / 21), 1],
    7: [-1, -0.830223896278566929872, -0.468848793470714213803772,
        0, 0.468848793470714213803772, 0.830223896278566929872, 1],
    8: [-1, -0.8717401485096066153375, -0.5917001814331423021445,
        -0.2092992179024788687687, 0.2092992179024788687687,
        0.5917001814331423021445, 0.8717401485096066153375, 1],
    9: [-1, -0.8997579954114601573124, -0.6771862795107377534459,
        -0.3631174638261781587108, 0, 0.3631174638261781587108,
        0.6771862795107377534459, 0.8997579954114601573124, 1],
    10: [-1, -0.9195339081664588138289, -0.7387738651055050750031,
         -0.4779249498104444956612, -0.1652789576663870246262,
         0.1652789576663870246262, 0.4779249498104444956612,
         0.7387738651055050750031, 0.9195339081664588138289, 1]
}

lgl_weights = {
    1: [2],
    2: [1, 1],
    3: [1 / 3, 4 / 3, 1 / 3],
    4: [1 / 6, 5 / 6, 5 / 6, 1 / 6],
    5: [1 / 10, 49 / 90, 32 / 45, 49 / 90, 1 / 10],
    6: [1 / 15, (14 - np.sqrt(7)) / 30, (14 + np.sqrt(7)) / 30,
        (14 + np.sqrt(7)) / 30, (14 - np.sqrt(7)) / 30, 1 / 15],
    7: [0.04761904761904761904762, 0.2768260473615659480107,
        0.4317453812098626234179, 0.487619047619047619048,
        0.4317453812098626234179, 0.2768260473615659480107,
        0.04761904761904761904762],
    8: [0.03571428571428571428571, 0.210704227143506039383,
        0.3411226924835043647642, 0.4124587946587038815671,
        0.4124587946587038815671, 0.3411226924835043647642,
        0.210704227143506039383, 0.03571428571428571428571],
    9: [0.02777777777777777777778, 0.1654953615608055250463,
        0.2745387125001617352807, 0.3464285109730463451151,
        0.3715192743764172335601, 0.3464285109730463451151,
        0.2745387125001617352807, 0.1654953615608055250463,
        0.02777777777777777777778],
    10: [0.02222222222222222222222, 0.1333059908510701111262,
         0.2248893420631264521195, 0.2920426836796837578756,
         0.3275397611838974566565, 0.3275397611838974566565,
         0.292042683679683757876, 0.224889342063126452119,
         0.133305990851070111126, 0.02222222222222222222222]
}


# noinspection PyTypeChecker
class Basis1D:
    def __init__(self, order):
        self.order = int(order)
        self.nodes = self.get_nodes()
        self.weights = self.get_weights()
        self.eigenvalues = self.set_eigenvalues()

        # Inverse vandermonde
        self.vandermonde_inverse = self.set_vandermonde_inverse()

        # Mass matrix and inverse
        self.mass = self.mass_matrix()
        self.d_mass = cp.asarray(self.mass)
        self.inv_m = self.inv_mass_matrix()
        self.face_mass = np.eye(self.order)[:, np.array([0, -1])]  # face mass, first and last columns of identity

        # Inner product arrays
        self.adv = self.advection_matrix()
        self.stf = self.adv.T

        # DG weak form arrays, numerical flux is first and last columns of inverse mass matrix
        # both are cupy arrays
        self.up = self.internal_flux()
        self.xi = cp.asarray(self.inv_m[:, np.array([0, -1])])
        # numpy array form
        # self.np_up = self.up.get()
        # self.np_xi = self.xi.get()

        # DG strong form array
        self.der = self.derivative_matrix()

    def get_nodes(self):
        nodes = lgl_nodes.get(self.order, "nothing")
        return nodes

    def get_weights(self):
        weights = lgl_weights.get(self.order, "nothing")
        return weights

    def set_eigenvalues(self):
        # Legendre-Lobatto "eigenvalues"
        eigenvalues = np.array([(s + 0.5) for s in range(self.order - 1)])

        # if self.order == 1:
        #    eigenvalues = 1 / 2
        return np.append(eigenvalues, (self.order - 1) / 2.0)

    def set_vandermonde_inverse(self):
        return np.array([[self.weights[j] * self.eigenvalues[s] * sp.legendre(s)(self.nodes[j])
                          for j in range(self.order)]
                         for s in range(self.order)])

    def mass_matrix(self):
        # Diagonal part
        approx_mass = np.diag(self.weights)

        # Off-diagonal part
        p = sp.legendre(self.order - 1)
        v = np.multiply(self.weights, p(self.nodes))
        a = -self.order * (self.order - 1) / (2 * (2 * self.order - 1))
        # calculate mass matrix
        return approx_mass + a * np.outer(v, v)

    def advection_matrix(self):
        adv = np.zeros((self.order, self.order))

        # Fill matrix
        for i in range(self.order):
            for j in range(self.order):
                adv[i, j] = self.weights[i] * self.weights[j] * sum(
                    self.eigenvalues[s] * sp.legendre(s)(self.nodes[i]) *
                    sp.legendre(s).deriv()(self.nodes[j]) for s in range(self.order))

        # Clean machine error
        adv[np.abs(adv) < 1.0e-15] = 0

        return adv

    def inv_mass_matrix(self):
        # Diagonal part
        approx_inv = np.diag(np.divide(1.0, self.weights))

        # Off-diagonal part
        p = sp.legendre(self.order - 1)
        v = p(self.nodes)
        b = self.order / 2
        # calculate inverse mass matrix
        return approx_inv + b * np.outer(v, v)

    def internal_flux(self):
        # Compute internal flux array
        up = np.zeros((self.order, self.order))
        for i in range(self.order):
            for j in range(self.order):
                up[i, j] = self.weights[j] * sum(
                    (2 * s + 1) / 2 * sp.legendre(s)(self.nodes[i]) *
                    sp.legendre(s).deriv()(self.nodes[j]) for s in range(self.order))

        # Clear machine errors
        up[np.abs(up) < 1.0e-10] = 0

        return cp.asarray(up)

    def derivative_matrix(self):
        der = np.zeros((self.order, self.order))

        for i in range(self.order):
            for j in range(self.order):
                der[i, j] = self.weights[j] * sum(
                    self.eigenvalues[s] * sp.legendre(s).deriv()(self.nodes[i]) *
                    sp.legendre(s)(self.nodes[j]) for s in range(self.order))

        # Clear machine errors
        der[np.abs(der) < 1.0e-15] = 0

        return der

    def fourier_transform_array(self, midpoints, J, wave_numbers):
        """
        Grid-dependent spectral coefficient matrix
        Needs grid quantities: Jacobian, wave numbers, nyquist number
        """
        # Spectral basis functions
        signs = np.sign(wave_numbers)
        signs[np.where(wave_numbers == 0)] = 1.0

        # Fourier-transformed modal basis ( (-1)^s accounts for scipy's failure to have negative spherical j argument )
        p_tilde = np.array([(signs ** s) * np.exp(-1j * np.pi / 2.0 * s) *
                            sp.spherical_jn(s, np.absolute(wave_numbers) / J) for s in range(self.order)])

        # Multiply by inverse Vandermonde transpose for fourier-transformed nodal basis
        ell_tilde = np.matmul(self.vandermonde_inverse.T, p_tilde)

        # Outer product with phase factors
        phase = np.exp(-1j * np.tensordot(midpoints, wave_numbers, axes=0))
        transform_array = np.multiply(phase[:, :, None], ell_tilde.T)

        # Put in order (resolution, nodes, modes)
        transform_array = np.transpose(transform_array, (0, 2, 1))

        # Return as cupy array
        return cp.asarray(transform_array)

    def interpolate_values(self, grid, arr):
        """ Determine interpolated values on a finer grid using the basis functions"""
        # Compute affine transformation per-element to isoparametric element
        xi = grid.J * (grid.arr_fine[1:-1, :] - grid.midpoints[:, None])
        # Legendre polynomials at transformed points
        ps = np.array([sp.legendre(s)(xi) for s in range(self.order)])
        # Interpolation polynomials at fine points
        ell = np.transpose(np.tensordot(self.vandermonde_inverse, ps, axes=([0], [0])), [1, 0, 2])
        # Compute interpolated values
        values = np.multiply(ell, arr[:, :, None]).sum(axis=1)
        return values


# noinspection PyTypeChecker
class Grid1D(Basis1D):
    def __init__(self, low, high, res, order, fine=False):  # spectrum=False,
        super().__init__(order)
        self.low = low
        self.high = high
        self.res = int(res)  # somehow gets non-int...
        self.res_ghosts = int(res + 2)  # resolution including ghosts
        # self.order = order  # basis.order

        # domain and element widths
        self.length = self.high - self.low
        self.dx = self.length / self.res

        # element Jacobian
        self.J = 2.0 / self.dx

        # full-grid quadrature weights
        self.quad_weights = cp.tensordot(cp.ones(self.res), cp.asarray(self.weights), axes=0)

        # arrays
        self.arr = np.zeros((self.res_ghosts, self.order))  # init grid array
        self.create_grid(self.nodes)  # fill grid array values
        self.arr_cp = cp.asarray(self.arr)  # array on device
        self.midpoints = np.array([(self.arr[i, -1] + self.arr[i, 0]) / 2.0 for i in range(1, self.res_ghosts - 1)])
        self.arr_max = np.amax(abs(self.arr))

        # velocity axis gets a positive/negative indexing slice
        self.one_negatives = cp.where(condition=self.arr_cp < 0, x=1, y=0)
        self.one_positives = cp.where(condition=self.arr_cp >= 0, x=1, y=0)

        # fine array
        if fine:
            fine_num = 25  # 200 for 1D poisson study
            self.arr_fine = np.array([np.linspace(self.arr[i, 0], self.arr[i, -1], num=fine_num)
                                      for i in range(self.res_ghosts)])

        # spectral coefficients
        self.k1 = 2.0 * np.pi / self.length  # fundamental mode
        # if spectrum:
        #     self.nyquist_number = 2.5 * self.length // self.dx  # mode number of nyquist frequency
        #     self.k1 = 2.0 * np.pi / self.length  # fundamental mode
        #     self.wave_numbers = self.k1 * np.arange(1 - self.nyquist_number, self.nyquist_number)
        #     self.d_wave_numbers = cp.asarray(self.wave_numbers)
        #     self.grid_phases = cp.asarray(np.exp(1j * np.tensordot(self.wave_numbers, self.arr[1:-1, :], axes=0)))
        #
        #     # Spectral matrices
        #     self.spectral_transform = basis.fourier_transform_array(self.midpoints, self.J, self.wave_numbers)

    def create_grid(self, nodes):
        """
        Initialize array of global coordinates (including ghost elements).
        """
        # shift to include ghost cells
        min_gs = self.low - self.dx
        max_gs = self.high  # + self.dx
        # nodes (iso-parametric)
        nodes = (np.array(nodes) + 1) / 2

        # element left boundaries (including ghost elements)
        xl = np.linspace(min_gs, max_gs, num=self.res_ghosts)

        # construct coordinates
        for i in range(self.res_ghosts):
            self.arr[i, :] = xl[i] + self.dx * nodes


class Variables:
    def __init__(self, params, grid, perturbation=True):  # om=0.9, om_pc=1.0, delta_n=0.01):
        # parameters
        self.params = params

        # if perturbation
        self.perturbation = perturbation

        # resolutions (no ghosts)
        self.x_res = grid.res_ghosts

        # orders
        self.x_ord = grid.order

        # array, initialize later
        self.arr_q = cp.zeros((3, self.x_res, self.x_ord))  # density, momentum density, and energy density

        # quad weights init
        # self.quad_weights = grid.quad_weights / grid.J

        # size0 = slice(grid.res_ghosts)

        self.boundary_slices = [
            # x-directed face slices [(left), (right)]
            [(grid.res_ghosts, 0), (grid.res_ghosts, -1)]]
        # Grid and sub-element axes
        self.grid_axis = np.array([0])
        self.sub_element_axis = np.array([1])

        # Initialize arrays
        self.initialize_gpu(grid)

    def initialize_gpu(self, grid):
        """
        Initialize distribution function as polar eigenfunction on GPU
        """
        # Grid indicators
        ix = cp.ones_like(grid.arr_cp)

        # Uniform states
        f0 = 1.0
        eig = 1.0  # desired velocity, 1.0
        f2 = 1.0
        self.arr_q[0, :, :] = f0 * ix
        self.arr_q[1, :, :] = eig * ix
        self.arr_q[2, :, :] = f2 * ix

        # Add perturbation
        if self.perturbation:
            amp = 0.05  # scalar mode amplitude
            amp_0 = amp
            amp_1 = amp * eig
            amp_2 = 0.5 * amp * (eig ** 2.0)
            self.arr_q[0, :, :] += amp_0 * cp.sin(grid.k1 * grid.arr_cp)
            self.arr_q[1, :, :] += amp_1 * cp.sin(grid.k1 * grid.arr_cp)
            self.arr_q[2, :, :] += amp_2 * cp.sin(grid.k1 * grid.arr_cp)

    def swap_ghost_cells(self):
        # left ghosts
        self.arr_q[:, 0, :] = cp.copy(self.arr_q[:, -2, :])
        # right ghosts
        self.arr_q[:, -1, :] = cp.copy(self.arr_q[:, 1, :])


# class Stages:
#     def __init__(self, time_order):
#         self.time_order = time_order
#         # array, initialize later
#         self.arr_q0 = None  # density
#         self.arr_q1 = None  # momentum density
#         self.arr_q2 = None  # energy density
#
#     def swap_ghost_cells(self):
#         # left ghosts
#         self.arr_q0[0, :] = cp.copy(self.arr_q0[-2, :])
#         self.arr_q1[0, :] = cp.copy(self.arr_q1[-2, :])
#         self.arr_q2[0, :] = cp.copy(self.arr_q2[-2, :])
#         # right ghosts
#         self.arr_q0[-1, :] = cp.copy(self.arr_q0[1, :])
#         self.arr_q1[-1, :] = cp.copy(self.arr_q1[1, :])
#         self.arr_q2[-1, :] = cp.copy(self.arr_q2[1, :])

import numpy as np
import cupy as cp


def basis_product(flux, basis_arr):
    return cp.tensordot(flux, basis_arr, axes=([[1], [1]]))


class DGFlux:
    def __init__(self, resolution, order, gamma=3.0):
        self.resolution = resolution + 2
        self.order = order
        self.gamma = gamma

        # boundary slices (not necessary in 1D)
        # self.boundary_slices = [(slice(resolution), 0), (resolution, -1)]

        # sound speed check
        self.max_speed = None

    def semi_discrete_rhs(self, variables, grid):  # , max_speeds):
        """
        Calculate the right-hand side of the semi-discrete equation
        """
        # Compute max speed for Lax-Friedrichs penalty term, element-wise
        # self.max_speed = max_speeds
        # Build flux vector
        pressure = (self.gamma - 1.0) * (variables[2, :, :] - 0.5 * variables[1, :, :] ** 2.0 / variables[0, :, :])
        flux0 = variables[1, :, :]
        flux1 = variables[1, :, :] ** 2.0 / variables[0, :, :] + pressure
        flux2 = (variables[2, :, :] + pressure) * variables[1, :, :] / variables[0, :, :]

        # Compute RHSs
        return cp.array([self.flux(variable_d=variables[0, :, :], flux_d=flux0, grid=grid),
                         self.flux(variable_d=variables[1, :, :], flux_d=flux1, grid=grid),
                         self.flux(variable_d=variables[2, :, :], flux_d=flux2, grid=grid)]) * grid.J

    def flux(self, variable_d, flux_d, grid):
        # compute internal flux
        internal = basis_product(flux=flux_d, basis_arr=grid.up)
        # compute numerical flux
        numerical = self.numerical_flux_lf(variable=variable_d, flux=flux_d, basis_arr=grid.xi)

        return internal - numerical

    def numerical_flux_lf(self, variable, flux, basis_arr):
        """
        Implements numerical flux as Lax-Friedrichs flux
        """
        # Allocate
        num_flux = cp.zeros((self.resolution, 2))

        # central flux, left face
        num_flux[:, 0] = -0.5 * (flux[:, 0] + cp.roll(flux[:, -1], shift=1, axis=0))
        # central flux, right face
        num_flux[:, -1] = 0.5 * (flux[:, -1] + cp.roll(flux[:, 0], shift=-1, axis=0))

        # penalty scalar
        # speed_left = cp.amax(cp.array([self.max_speed[:, 0],
        #                                cp.roll(self.max_speed[:, -1],
        #                                        shift=1, axis=0)]),
        #                      axis=0)
        # speed_right = cp.amax(cp.array([self.max_speed[:, -1],
        #                                 cp.roll(self.max_speed[:, 0],
        #                                         shift=-1, axis=0)]),
        #                       axis=0)

        # penalty flux, left face
        # num_flux[:, 0] += -0.5 * cp.multiply(speed_left, variable[:, 0] - cp.roll(variable[:, -1], shift=1, axis=0))
        # num_flux[:, -1] += 0.5 * cp.multiply(speed_right, variable[:, -1] - cp.roll(variable[:, 0], shift=-1, axis=0))

        return basis_product(flux=num_flux, basis_arr=basis_arr)

    def dg_jacobian(self, irk_stages, grid):
        """
        Computes Jacobian of DG system
        """
        # Euler system jacobian
        flux_jacobian = self.flux_jacobian(irk_stages)

        # Compute DG jacobian due to elemental internal fluxes
        dg_jacobian = cp.einsum('ij,klmis->mikjls', grid.up, flux_jacobian)
        dg_jacobian = cp.einsum('abidjs,ag->abigdjs', dg_jacobian, cp.eye(irk_stages.shape[1]))

        # Element boundary fluxes:
        # Jacobian contributions: Left element edge boundary flux
        part1 = cp.einsum('ag,ijas->agijs',
                          cp.eye(irk_stages.shape[1]),
                          flux_jacobian[:, :, :, 0, :])
        dg_jacobian[:, :, :, :, 0, :, :] += 0.5 * cp.einsum('b,agijs->abigjs',
                                                         grid.xi[:, 0],
                                                         part1)
        part1 = cp.roll(cp.einsum('ag,ijas->agijs',
                                  cp.eye(irk_stages.shape[1]),
                                  flux_jacobian[:, :, :, -1, :]),
                        shift=1, axis=0)
        dg_jacobian[:, :, :, :, -1, :, :] += 0.5 * cp.einsum('b,agijs->abigjs',
                                                          grid.xi[:, 0],
                                                          part1)
        # Jacobian contributions: Right element edge boundary flux
        part1 = cp.einsum('ag,ijas->agijs',
                          cp.eye(irk_stages.shape[1]),
                          flux_jacobian[:, :, :, -1])
        dg_jacobian[:, :, :, :, -1, :, :] -= 0.5 * cp.einsum('b,agijs->abigjs',
                                                          grid.xi[:, -1],
                                                          part1)
        part1 = cp.roll(cp.einsum('ag,ijas->agijs',
                                  cp.eye(irk_stages.shape[1]),
                                  flux_jacobian[:, :, :, 0, :]),
                        shift=-1, axis=0)
        dg_jacobian[:, :, :, :, 0, :, :] -= 0.5 * cp.einsum('b,agijs->abigjs',
                                                         grid.xi[:, -1],
                                                         part1)

        return dg_jacobian

    def flux_jacobian(self, variables):
        """
        Computes Euler flux Jacobian
        """
        # Initialize
        flux_jacobian = cp.zeros((variables.shape[0], variables.shape[0],
                                  variables.shape[1], variables.shape[2], variables.shape[3]))
        # Compute pressure and velocity
        velocity = variables[1, :, :, :] / variables[0, :, :, :]
        pressure = (self.gamma - 1.0) * (variables[2, :, :, :] -
                                         0.5 * variables[1, :, :, :] ** 2.0 / variables[0, :, :, :])
        enthalphy = (variables[2, :, :, :] + pressure) / variables[0, :, :, :]
        # Fill elements
        flux_jacobian[0, 1, :, :, :] = 1.0
        flux_jacobian[1, 0, :, :, :] = 0.5 * (self.gamma - 3.0) * (velocity ** 2.0)
        flux_jacobian[1, 1, :, :, :] = (3.0 - self.gamma) * velocity
        flux_jacobian[1, 2, :, :, :] = self.gamma - 1.0
        flux_jacobian[2, 0, :, :, :] = 0.5 * (self.gamma - 1.0) * (velocity ** 3.0) - velocity * enthalphy
        flux_jacobian[2, 1, :, :, :] = enthalphy - (self.gamma - 1.0) * (velocity ** 2.0)
        flux_jacobian[2, 2, :, :, :] = self.gamma * velocity

        return flux_jacobian

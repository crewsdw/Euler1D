import numpy as np
import cupy as cp
from matplotlib import pyplot as plt

import basis as b
import time as timer
import newton as newton

# Courant numbers for RK-DG stability from Cockburn and Shu 2001, [time_order][space_order-1]
courant_numbers = {
    1: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    2: [1.0, 0.333],
    3: [1.256, 0.409, 0.209, 0.130, 0.089, 0.066, 0.051, 0.040, 0.033],
    4: [1.392, 0.464, 0.235, 0.145, 0.100, 0.073, 0.056, 0.045, 0.037],
    5: [1.608, 0.534, 0.271, 0.167, 0.115, 0.085, 0.065, 0.052, 0.042],
    6: [1.776, 0.592, 0.300, 0.185, 0.127, 0.093, 0.072, 0.057, 0.047],
    7: [1.977, 0.659, 0.333, 0.206, 0.142, 0.104, 0.080, 0.064, 0.052],
    8: [2.156, 0.718, 0.364, 0.225, 0.154, 0.114, 0.087, 0.070, 0.057]
}

nonlinear_ssp_rk_switch = {
    2: [[1 / 2, 1 / 2, 1 / 2]],
    3: [[3 / 4, 1 / 4, 1 / 4],
        [1 / 3, 2 / 3, 2 / 3]]
}


class Stepper:
    def __init__(self, time_order, space_order, stop_time, write_time, gamma=3.0, explicit=False):
        self.time_order = time_order
        self.space_order = space_order
        self.coefficients = self.get_nonlinear_coefficients()
        self.courant = None
        if explicit:
            self.courant = self.get_courant_number()
        self.gamma = gamma

        # Simulation time init
        self.time = 0
        self.dt = None
        self.steps_counter = 0
        self.write_counter = 0
        self.stop_time = stop_time
        self.write_time = write_time

        # max-speeds
        self.max_speeds = None

    def get_nonlinear_coefficients(self):
        return np.array(nonlinear_ssp_rk_switch.get(self.time_order, "nothing"))

    def get_courant_number(self):
        return courant_numbers.get(self.time_order)[self.space_order - 1]

    def main_loop_implicit(self, variables, grid, dg_flux, irk, dt):
        self.dt = dt
        # Loop while time is less than final time
        t0 = timer.time()
        while self.time < self.stop_time:
            # Perform IRK update
            self.implicit_rk(variables=variables, grid=grid, dg_flux=dg_flux, irk=irk)
            # Update time and steps counter
            self.time += self.dt
            self.steps_counter += 1
            # Print update
            if self.time > self.write_counter * self.write_time:
                print('\nI made it through step ' + str(self.steps_counter))
                self.write_counter += 1
                print('The simulation time is {:0.3e}'.format(self.time))
                print('The time-step is {:0.3e}'.format(self.dt))
                print('Time since start is ' + str((timer.time() - t0) / 60.0) + ' minutes')
        print('\nFinal time reached')
        print('Total steps were ' + str(self.steps_counter))

    def implicit_rk(self, variables, grid, dg_flux, irk):
        # Visualize stages
        plt.figure()
        plt.plot(grid.arr[1:-1, :].flatten(), variables.arr_q[0, 1:-1, :].flatten().get(),
                 'o--', label='q_0')
        # A terrible guess
        stages = cp.ones((self.time_order,
                          variables.arr_q.shape[0], variables.arr_q.shape[1], variables.arr_q.shape[2]))
        # dg_flux.dg_jacobian(variables.arr_q, grid)
        rhs = newton.newton_irk(q0=variables.arr_q, guess=stages, dt=self.dt,
                                threshold=1.0e-10, max_iterations=100, irk=irk, grid=grid, dg_flux=dg_flux)
        # Extract stages from solved RHS
        stages = variables.arr_q[None, :, :, :] + self.dt * cp.tensordot(irk.rk_matrix, rhs, axes=([1], [0]))
        # IRK time-step update, stage combination
        variables.arr_q += 0.5 * self.dt * np.tensordot(irk.weights_device, rhs, axes=([0], [0]))

        # Continue plot
        for i in range(self.time_order):
            plt.plot(grid.arr[1:-1, :].flatten(), stages[i, 0, 1:-1, :].flatten().get(),
                     'o--', label='rk stage ' + str(i))
        plt.plot(grid.arr[1:-1, :].flatten(), variables.arr_q[0, 1:-1, :].flatten().get(),
                 'o--', label='q_1')
        plt.legend(loc='best')
        plt.show()
        # quit()

    def main_loop_explicit(self, variables, grid, dg_flux):
        # Loop while time is less than final time
        t0 = timer.time()
        # adapt time-step
        print('\nInitializing time-step...')
        self.get_max_speeds(variables=variables)
        self.adapt_time_step(dx=grid.dx)
        while self.time < self.stop_time:
            # Perform RK update
            self.nonlinear_ssp_rk(variables=variables, grid=grid, dg_flux=dg_flux)
            # Update time and steps counter
            self.time += self.dt
            self.steps_counter += 1
            # Print update
            if self.time > self.write_counter * self.write_time:
                print('\nI made it through step ' + str(self.steps_counter))
                self.write_counter += 1
                print('The simulation time is {:0.3e}'.format(self.time))
                print('The time-step is {:0.3e}'.format(self.dt))
                print('Time since start is ' + str((timer.time() - t0) / 60.0) + ' minutes')

        print('\nFinal time reached')
        print('Total steps were ' + str(self.steps_counter))

    def nonlinear_ssp_rk(self, variables, grid, dg_flux):
        # Sync ghost cells
        variables.swap_ghost_cells()
        # Set up stages
        t_shape = tuple([self.time_order] + [size for size in variables.arr_q.shape])
        q_stages = cp.zeros(t_shape)

        # Compute first RK stage
        semi_discrete = dg_flux.semi_discrete_rhs(variables=variables.arr_q, grid=grid, max_speeds=self.max_speeds)
        q_stages[0, :, 1:-1, :] = variables.arr_q[:, 1:-1, :] + self.dt * semi_discrete[:, 1:-1, :]

        # Compute further stages
        for i in range(1, self.time_order):
            # swap, ghosts
            q_stages[i - 1, :, 0, :] = cp.copy(q_stages[i - 1, :, -2, :])
            q_stages[i - 1, :, -1, :] = cp.copy(q_stages[i - 1, :, 1, :])
            # Next stage
            semi_discrete = dg_flux.semi_discrete_rhs(variables=q_stages[i - 1, :, :, :],
                                                      grid=grid, max_speeds=self.max_speeds)
            # Update stage
            q_stages[i, :, 1:-1, :] = (self.coefficients[i - 1, 0] * variables.arr_q +
                                       self.coefficients[i - 1, 1] * q_stages[i - 1, :, :, :] +
                                       self.coefficients[i - 1, 2] * self.dt * semi_discrete)[:, 1:-1, :]
        # Complete update
        variables.arr_q[:, 1:-1, :] = q_stages[-1, :, 1:-1, :]

        # Adapt time-step
        self.get_max_speeds(variables=variables)
        self.adapt_time_step(dx=grid.dx)

    def get_max_speeds(self, variables):
        pressure = (self.gamma - 1.0) * (variables.arr_q[2, :, :] -
                                         0.5 * variables.arr_q[1, :, :] ** 2.0 / variables.arr_q[0, :, :])
        # print(pressure)
        sound = cp.sqrt(self.gamma * pressure / variables.arr_q[0, :, :])
        velocity = variables.arr_q[1, :, :] / variables.arr_q[0, :, :]
        self.max_speeds = cp.abs(velocity) + sound

    def adapt_time_step(self, dx):
        self.dt = 0.95 * self.courant / (cp.amax(self.max_speeds) / dx)
        # print(self.dt)

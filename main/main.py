import numpy as np
import cupy as cp
import basis as bs
import fluxes as fx
import timestep as ts
import irk_gl as irk
import neural_net as nn

import matplotlib.pyplot as plt

import tensorflow as tf

# Flags
plot_IC = True

# Parameters
order = 3
time_order = 3  # 3
res = 20
# folder = '..\\data\\'
# filename = 'euler_test'

# geometry
L = 2.0 * np.pi
low = -0.5 * L
high = 0.5 * L

# time info
dt = 0.3
stop_time = 1.0
write_time = 1.0

# basis
print('\nSetting up basis and grid...')
grid = bs.Grid1D(low=low, high=high, res=res, order=order)
print('\nInitializing conserved variables...')
variables = bs.Variables(params=[], grid=grid)

# Time basis
IRK = irk.IRK(order=time_order)
IRK.build_matrix()

# Initial-condition
IC = cp.copy(variables.arr_q)

if plot_IC:
    plt.figure()
    plt.plot(grid.arr[1:-1, :].flatten(), variables.arr_q[0, 1:-1, :].flatten().get(), 'o--', label='q0')
    plt.plot(grid.arr[1:-1, :].flatten(), variables.arr_q[2, 1:-1, :].flatten().get(), 'o--', label='q2')
    plt.xlabel('x')
    plt.ylabel('q')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()

    plt.figure()
    plt.plot(grid.arr[1:-1, :].flatten(), variables.arr_q[1, 1:-1, :].flatten().get(), 'o--', label='q1')
    plt.xlabel('x')
    plt.ylabel('q')
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()

    plt.show()

# Try predicting RK output stages with neural net
# Net parameters
parameters = [dt, 3.0]  # gamma = 3
# neurons = nodes
activation = 'elu'
optimizer = 'adam'
epochs = 10000

xtf = tf.convert_to_tensor(grid.arr[1:-1, :].flatten(), dtype=tf.float32)
xtf = tf.reshape(xtf, (xtf.shape[0], 1))
# u0 = np.asarray(variables.arr_q[:, 1:-1, :].flatten().get())
utf = tf.transpose(tf.reshape(tf.convert_to_tensor(variables.arr_q[:, 1:-1, :].get(),
                                                   dtype=tf.float32),
                              (3, xtf.shape[0])),
                   perm=(1, 0))
net = nn.NeuralNetEuler(parameters=parameters, grid=grid, irk=IRK,
                        neurons=xtf.shape[0], activation=activation)
net.compile(optimizer=optimizer, loss=net.loss)
net.fit(xtf, utf, epochs=epochs, shuffle=True)
out = net.predict(xtf, batch_size=xtf.shape[0])

# Outputs
print(out.shape)
stages = out[0, :, :, :]  # .reshape(grid.arr[1:-1, :].shape[0], grid.arr.shape[1], 3, time_order+1)
print(stages.shape)
rhs = out[1, :, :, :]  # .reshape(grid.arr[1:-1, :].shape[0], grid.arr.shape[1], 3, time_order+1)
u0_net = (stages - dt * tf.einsum('ijk,lk->ijl', rhs, IRK.rk_matrix_tf32)).numpy()

# Plot
plt.figure()
plt.plot(grid.arr[1:-1, :].flatten(), IC[0, 1:-1, :].flatten().get(), 'o--', label='q_0')
for i in range(time_order):
    plt.plot(grid.arr[1:-1, :].flatten(), stages[:, 0, i].flatten(),
             'o--', label='rk stage ' + str(i))
plt.plot(grid.arr[1:-1, :].flatten(), stages[:, 0, -1].flatten(),
         'o--', label='q_1')
plt.legend(loc='best')

plt.figure()
plt.plot(grid.arr[1:-1, :].flatten(), IC[0, 1:-1, :].flatten().get(), 'o--', label='q_0')
for i in range(time_order):
    plt.plot(grid.arr[1:-1, :].flatten(), u0_net[:, 0, i].flatten(),
             'o--', label='combined stage ' + str(i))
plt.plot(grid.arr[1:-1, :].flatten(), u0_net[:, 0, -1].flatten(),
         'o--', label='q_1')
plt.legend(loc='best')

plt.show()

quit()

# Set up fluxes
print('\nSetting up fluxes...')
dg_flux = fx.DGFlux(resolution=res, order=order)

# Set up time-step
print('\nSetting up time-stepper...')
stepper = ts.Stepper(time_order=time_order, space_order=order, stop_time=stop_time, write_time=write_time)

# Time-step loop
print('\nBeginning main loop...')
stepper.main_loop_implicit(variables=variables, grid=grid, dg_flux=dg_flux, irk=IRK, dt=2.5e-2)
# stepper.main_loop_explicit(variables=variables, grid=grid, dg_flux=dg_flux)

# Visualize post
plt.figure()
plt.plot(grid.arr[1:-1, :].flatten(), IC[0, 1:-1, :].flatten().get(), 'o--', label='IC, q0')
plt.plot(grid.arr[1:-1, :].flatten(), IC[2, 1:-1, :].flatten().get(), 'o--', label='IC, q2')

plt.plot(grid.arr[1:-1, :].flatten(), variables.arr_q[0, 1:-1, :].flatten().get(), 'o--', label='stop, q0')
plt.plot(grid.arr[1:-1, :].flatten(), variables.arr_q[2, 1:-1, :].flatten().get(), 'o--', label='stop, q2')
plt.xlabel('x')
plt.ylabel('q')
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()

plt.figure()
plt.plot(grid.arr[1:-1, :].flatten(), IC[1, 1:-1, :].flatten().get(), 'o--', label='IC, q1')
plt.plot(grid.arr[1:-1, :].flatten(), variables.arr_q[1, 1:-1, :].flatten().get(), 'o--', label='stop, q1')
plt.xlabel('x')
plt.ylabel('q')
plt.grid(True)
plt.legend(loc='best')
plt.tight_layout()


plt.show()

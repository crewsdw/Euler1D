import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


class NeuralNetEuler(tf.keras.Model):
    def __init__(self, parameters, grid, irk, neurons, activation='elu'):
        super(NeuralNetEuler, self).__init__()
        # problem parameters
        self.dt = parameters[0]
        self.gamma = parameters[1]

        # runge-kutta coefficient matrix and weights
        self.irk = irk

        # net parameters
        self.neurons = neurons
        self.activation = activation

        # boundary
        self.bc = tf.reshape(tf.constant([-np.pi, np.pi], dtype=tf.float32), (2, 1))

        # net structure
        self.net = tf.keras.Sequential([
            # tf.keras.layers.InputLayer(input_shape=(self.neurons, 1)),
            tf.keras.layers.Dense(self.neurons, activation=self.activation),
            tf.keras.layers.Dense(self.neurons, activation=self.activation),
            # tf.keras.layers.Dense(self.neurons, activation=self.activation),
            # tf.keras.layers.Dense(self.neurons, activation=self.activation),
            # tf.keras.layers.Dense(self.neurons, activation=self.activation),
            tf.keras.layers.Dense((self.irk.order + 1) * 3, activation='linear'),
            tf.keras.layers.Reshape((3, self.irk.order + 1))
        ])

    @tf.autograph.experimental.do_not_convert
    def call(self, x, training=None, mask=None):
        u1 = self.net(x)

        # Compute flux vector of net output
        pressure = (self.gamma - 1.0) * (u1[:, 2, :] - 0.5 * u1[:, 1, :] ** 2.0 / u1[:, 0, :])
        flux0 = u1[:, 1, :]
        flux1 = u1[:, 1, :] ** 2.0 / u1[:, 0, :] + pressure
        flux2 = (u1[:, 2, :] + pressure) * u1[:, 1, :] / u1[:, 0, :]

        # Compute spatial gradients of flux
        flux0_x, flux1_x, flux2_x = [], [], []
        for i in range(self.irk.order + 1):
            flux0_x.append(tf.gradients(flux0[:, i], x)[0])
            flux1_x.append(tf.gradients(flux1[:, i], x)[0])
            flux2_x.append(tf.gradients(flux2[:, i], x)[0])
        
        a = tf.stack(flux0_x)[:, :, 0]  # tf.reshape(tf.stack(u_x), (self.irk.order+1, self.neurons))
        b = tf.stack(flux1_x)[:, :, 0]  # tf.reshape(tf.stack(u_xx), (self.irk.order+1, self.neurons))
        c = tf.stack(flux2_x)[:, :, 0]

        flux_x = tf.transpose(tf.stack([a, b, c]), perm=(2, 0, 1))

        return tf.stack([u1, flux_x])

    # Loss function
    @tf.autograph.experimental.do_not_convert
    def loss(self, u0_true, u1_prediction):
        u1 = u1_prediction[0, :, :, :]
        rhs = -u1_prediction[1, :, :, :]

        # Objective function: match initial condition
        # print(u1)
        # print(rhs)
        # print(self.irk.rk_matrix_tf32.shape)
        # print(tf.tensordot(rhs, self.irk.rk_matrix_tf32, axes=([2], [0])))
        # quit()
        # u0 = u1 - self.dt * tf.transpose(tf.matmul(rhs, self.irk.rk_matrix_tf32), perm=(1, 0, 2))
        # u00 = u1[:, 0, :] - self.dt * tf.transpose(tf.matmul(rhs[:, 0, :], self.irk.rk_matrix_tf32), perm=(1, 0))
        # u01 = u1[:, 1, :] - self.dt * tf.transpose(tf.matmul(rhs[:, 1, :], self.irk.rk_matrix_tf32), perm=(1, 0))
        # u02 = u1[:, 2, :] - self.dt * tf.transpose(tf.matmul(rhs[:, 2, :], self.irk.rk_matrix_tf32), perm=(1, 0))
        # print(u1)
        # print(u00)
        # print(tf.stack([u00, u01, u02]))
        # quit()
        # u0 = u1 - self.dt * tf.tensordot(rhs, self.irk.rk_matrix_tf32, axes=([2], [1]))
        u0 = u1 - self.dt * tf.einsum('ijk,lk->ijl', rhs, self.irk.rk_matrix_tf32)

        # Error
        error = u0_true[:, :, None] - u0
        sqr_error = K.square(error)
        mean_sqr_error = K.mean(sqr_error)

        # Periodic boundary loss
        boundaries = self.call(self.bc)[0, :, :, :]
        boundary_error = K.mean(K.square(boundaries[0, :, :] - boundaries[1, :, :]))

        return mean_sqr_error + boundary_error

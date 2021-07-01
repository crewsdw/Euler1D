import cupy as cp
import cupy.linalg as linalg


# IRK newton solve
def newton_irk(q0, guess, dt, threshold, max_iterations, irk, grid, dg_flux):
    size = guess.shape[0] * guess.shape[1] * (guess.shape[2]-2) * guess.shape[3]
    # Compute RHS of guess
    rhs = cp.array([dg_flux.semi_discrete_rhs(variables=guess[i, :, :, :], grid=grid) for i in range(guess.shape[0])])

    # jac = dg_flux.dg_jacobian(variables=guess, grid=grid)

    def error(rhs_in):
        # Variable advance
        q_out = q0 + dt * cp.tensordot(irk.rk_matrix, rhs_in, axes=([1], [0]))
        # Swap ghosts
        q_out[:, :, 0, :] = q_out[:, :, -2, :]
        q_out[:, :, -1, :] = q_out[:, :, 1, :]
        rhs_out = cp.array([dg_flux.semi_discrete_rhs(variables=q_out[i, :, :, :], grid=grid)
                            for i in range(guess.shape[0])])
        # Compute error, exclude ghosts
        err = rhs_in[:, :, 1:-1, :] - rhs_out[:, :, 1:-1, :]
        # print(err)
        # quit()
        return err, cp.sqrt(cp.square(err).sum())

    # Newton iteration
    itr = 0
    err_arr, l2_err = error(rhs)
    while l2_err > threshold and itr < max_iterations:
        # Variable advance
        q_out = q0 + dt * cp.tensordot(irk.rk_matrix, rhs, axes=([1], [0]))
        # Swap ghosts
        q_out[:, :, 0, :] = q_out[:, :, -2, :]
        q_out[:, :, -1, :] = q_out[:, :, 1, :]

        # DG Jacobian
        jacobian = dg_flux.dg_jacobian(irk_stages=cp.transpose(q_out, axes=(1, 2, 3, 0)), grid=grid)
        # IRK Jacobian
        irk_jacobian = (cp.transpose(cp.tensordot(cp.tensordot(cp.eye(irk.order), cp.eye(q0.shape[1]-2), axes=0),
                                                  cp.tensordot(cp.eye(q0.shape[2]), cp.eye(q0.shape[0]), axes=0),
                                                  axes=0),
                                     axes=(0, 2, 4, 6, 1, 3, 5, 7))
                        - dt * cp.einsum('xz,abigdjx->xabizgdj', irk.rk_matrix, jacobian[1:-1, :, :, 1:-1, :, :, :]))

        # Flatten and solve
        solution = linalg.solve(irk_jacobian.reshape(size, size), err_arr.reshape(size)).reshape(err_arr.shape)

        # Iterate
        damping = 0.9
        rhs[:, :, 1:-1, :] -= damping * solution
        itr += 1

        # Recompute error
        err_arr, l2_err = error(rhs)
        print('In iteration ' + str(itr) + ' error is %.3e' % l2_err)

        if itr >= max_iterations:
            print('Did not converge by max iterations')
            return rhs

    print('Newton iteration took ' + str(itr) + ' tries, with error %.3e' % l2_err)
    return rhs

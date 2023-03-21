"""
MIT License

Copyright (c) 2023 Elena Oikonomou

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import jax.numpy as jnp


class DDP:
    def __init__(self, n, m, f, f_x, f_u, f_xx, f_uu, f_xu, f_ux, l_x, l_u, l_xx, l_uu, lf_x, lf_xx, cost):
        self.n = n              # Number of states
        self.m = m              # Number of controls
        self.f = f              # Discrete dynamics
        self.f_x = f_x
        self.f_u = f_u
        self.f_xx = f_xx
        self.f_uu = f_uu
        self.f_xu = f_xu
        self.f_ux = f_ux
        self.l_x = l_x
        self.l_u = l_u
        self.l_xx = l_xx
        self.l_uu = l_uu
        self.lf_x = lf_x
        self.lf_xx = lf_xx
        self.cost = cost        # Objective function cost

        self.N = None           # Number of timesteps
        self.eps = 1e-3         # Tolerance
        self.max_iter = 800     # Maximum iterations for DDP
        self.max_iter_reg = 10  # Maximum iterations for regularization

    def run_DDP(self, x0, u_traj, N):
        """Runs the iLQR or DDP algorithm.

        Inputs:
          - x0(np.ndarray):     The initial state            [nx1]
          - u_traj(np.ndarray): The initial control solution [mxN]
          - N(int):             Simulation timesteps
        Returns:
          - x_traj(np.ndarray): The states trajectory        [nxN]
          - u_traj(np.ndarray): The controls trajectory      [mxN]
          - J(float):           The optimal cost
        """
        print("DDP beginning..")
        self.N = N

        # Initial Rollout
        x_traj = np.zeros((self.n, self.N))
        x_traj[:, 0] = x0
        for k in range(self.N-1):
            x_traj[:, k+1] = self.f(x_traj[:, k], u_traj[:, k])
        J = self.cost(x_traj, u_traj)

        # Initialize DDP matrices
        p = np.ones((self.n, self.N))
        P = np.zeros((self.n, self.n, self.N))
        d = np.ones(self.N-1)
        K = np.zeros((self.m, self.n, self.N-1))

        itr = 0
        prev_err = np.inf
        err_diff = np.inf
        while np.linalg.norm(d, np.inf) > self.eps and itr < self.max_iter and err_diff > 1e-6:
            # Backward Pass
            DJ, p, P, d, K = self.backward_pass(p, P, d, K, x_traj, u_traj)

            # Forward rollout with line search
            x_new, u_new, Jn = self.rollout_with_linesearch(x_traj, u_traj, d, K, J, DJ)

            # Update values
            J = Jn
            x_traj = x_new
            u_traj = u_new
            itr += 1
            err_diff = abs(np.linalg.norm(d, np.inf) - prev_err)
            prev_err = np.linalg.norm(d, np.inf)

            if itr%10 == 0:
                print("Iteration: {}, J = {}".format(itr, J))

        print("\nDDP took: {} iterations".format(itr))
        return x_traj, u_traj, J

    def backward_pass(self, p, P, d, K, x_traj, u_traj):
        """Performs a backward pass to update values."""
        DJ = 0.0
        p[:, self.N-1] = self.lf_x(x_traj[:, self.N-1])
        P[:, :, self.N-1] = self.lf_xx()

        for k in range(self.N-2, -1, -1):

            # Compute derivatives
            A = self.f_x(x_traj[:, k], u_traj[:, k])
            B = self.f_u(x_traj[:, k], u_traj[:, k])

            gx = self.l_x(x_traj[:, k]) + A.T@p[:, k+1]  # (n,)
            gu = self.l_u(u_traj[:, k]) + B.T@p[:, k+1]  # (m,)

            # iLQR (Gauss-Newton) version
            # ------------------------------------------------------------------
            Gxx = self.l_xx() + A.T@P[:, :, k+1]@A        # nxn
            Guu = self.l_uu() + B.T@P[:, :, k+1]@B        # mxm
            Gxu = A.T@P[:, :, k+1]@B                      # nxm
            Gux = B.T@P[:, :, k+1]@A                      # mxn

            # DDP (full Newton) version
            # ------------------------------------------------------------------
            # Ax = self.f_xx(x_traj[:, k], u_traj[:, k])  # nnxn
            # Bx = self.f_ux(x_traj[:, k], u_traj[:, k])  # nxn
            # Au = self.f_xu(x_traj[:, k], u_traj[:, k])  # nnxm
            # Bu = self.f_uu(x_traj[:, k], u_traj[:, k])  # (n,)
            #
            # Gxx = self.l_xx() + A.T@P[:,:,k+1]@A + jnp.kron(p[:,k+1].T, jnp.eye(self.n))@self.comm_mat(self.n, self.n)@Ax  # nxn
            # Guu = self.l_uu() + B.T@P[:,:,k+1]@B + jnp.kron(p[:,k+1].T, jnp.eye(self.m))@self.comm_mat(self.n, self.m)@Bu  # mxm
            # Gxu = A.T@P[:,:,k+1]@B + jnp.kron(p[:,k+1].T, jnp.eye(self.n))@self.comm_mat(self.n, self.n)@Au                # nxm
            # Gux = B.T@P[:,:,k+1]@A + jnp.kron(p[:,k+1].T, jnp.eye(self.m))@self.comm_mat(self.n, self.m)@Bx                # mxn
            #
            # # Regularization
            # beta = 0.1
            # G = np.block([[Gxx, Gxu],
            #               [Gux, Guu]])
            # iter_reg = 0
            # while not self.is_pos_def(G) and iter_reg < self.max_iter_reg:
            #     Gxx += beta*A.T@A
            #     Guu += beta*B.T@B
            #     Gxu += beta*A.T@B
            #     Gux += beta*B.T@A
            #     beta = 2*beta
            #     # print("regularizing G")
            #     iter_reg += 1
            # # ------------------------------------------------------------------

            d[k], _, _, _ = jnp.linalg.lstsq(Guu, gu, rcond=None)
            K[:, :, k], _, _, _ = jnp.linalg.lstsq(Guu, Gux, rcond=None)
            p[:, k] = gx - K[:, :, k].T@gu + (K[:, :, k].T@Guu*d[k]).reshape(4,) - (Gxu*d[k]).reshape(4,)
            P[:, :, k] = Gxx + K[:, :, k].T@Guu@K[:, :, k] - Gxu@K[:, :, k] - K[:, :, k].T@Gux

            DJ += gu.T*d[k]

        return DJ, p, P, d, K

    @staticmethod
    def is_pos_def(A):
        """Check if matrix A is positive definite.

        If symmetric and has Cholesky decomposition -> p.d.
        """
        if np.allclose(A, A.T, rtol=1e-04, atol=1e-04):  # Ensure it is symmetric
            try:
                np.linalg.cholesky(A)
                return True
            except np.linalg.LinAlgError:
                return False
        else:
            return False

    def rollout(self, x_traj, u_traj, d, K, a):
        """Forward rollout."""
        x_new = np.zeros((self.n, self.N))
        u_new = np.zeros((self.m, self.N-1))
        x_new[:, 0] = x_traj[:, 0]
        for k in range(self.N-1):
            u_new[:, k] = u_traj[:, k] - a*d[k] - np.dot(K[:, :, k], x_new[:, k] - x_traj[:, k])
            x_new[:, k+1] = self.f(x_new[:, k], u_new[:, k])

        J_new = self.cost(x_new, u_new)
        return x_new, u_new, J_new

    def rollout_with_linesearch(self, x_traj, u_traj, d, K, J, DJ):
        """Forward rollout with linesearch to find best step size."""
        a = 1.0     # Step size
        b = 1e-2    # Armijo tolerance
        x_new, u_new, Jn = self.rollout(x_traj, u_traj, d, K, a)

        while Jn > (J - b*a*DJ):
            a = 0.5*a
            x_new, u_new, Jn = self.rollout(x_traj, u_traj, d, K, a)

        return x_new, u_new, Jn

    @staticmethod
    def comm_mat(m, n):
        """Commutation matrix.

        Used to transform the vectorized form of a matrix into the vectorized
        form of its transpose.
        Inputs:
          - m(int): Number of rows
          - n(int): Number of columns
        """
        w = np.arange(m * n).reshape((m, n), order="F").T.ravel(order="F")
        return np.eye(m * n)[w, :]

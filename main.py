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

from jax.config import config
config.update("jax_enable_x64", True)   # enable float64 types for accuracy

import jax.numpy as jnp

from model import Model
from ddp import DDP
from dynamical_sys import acrobot_dynamics, cartpole_dynamics
from plotter import CartpolePlotter, AcrobotPlotter, plot_trajectories


def acrobot_data():
    """Simulation parameters for the Acrobot."""
    N = 201                                         # Timesteps
    Ts = 0.05                                       # Discretization step
    n = 4                                           # Number of states
    m = 1                                           # Number of controls
    Q = jnp.diag(jnp.array([1.0, 1.0, 0.1, 0.1]))   # State cost weight
    Qf = 100*jnp.eye(n)                             # Final state cost weight
    R = 0.01*jnp.eye(m)                             # Controls cost weight
    x_goal = jnp.array([jnp.pi/2, 0, 0, 0])         # Goal state
    x0 = jnp.array([-jnp.pi/2, 0.0, 0.0, 0.0])      # Initial state
    u_traj = 0.001*jnp.ones((1, N - 1))             # Initial sol (different from 0 to avoid issues due to symmetry)
    return n, m, x_goal, Ts, acrobot_dynamics, Q, Qf, R, x0, u_traj, N


def cartpole_data():
    """Simulation parameters for the Cartpole."""
    N = 51                                          # Timesteps
    Ts = 0.1                                        # Discretization step
    n = 4                                           # Number of states
    m = 1                                           # Number of controls
    Q = jnp.eye(n)                                  # State cost weight
    Qf = 100*jnp.eye(n)                             # Final state cost weight
    R = 0.1*jnp.eye(m)                              # Controls cost weight
    x_goal = jnp.array([0, jnp.pi, 0, 0])           # Goal state
    x0 = jnp.array([0.0, 0.0, 0.0, 0.0])            # Initial state
    u_traj = 0.001*jnp.ones((1, N-1))               # Initial sol (different from 0 to avoid issues due to symmetry)
    return n, m, x_goal, Ts, cartpole_dynamics, Q, Qf, R, x0, u_traj, N


def main():
    """
    Non-linear trajectory optimization via iLQR/DDP for the Acrobot and
    Cartpole dynamical systems.
    """

    # Select system: Acrobot or Cartpole
    dyn_sys = "Cartpole"

    # Load simulation parameters
    if dyn_sys == "Acrobot":
        n, m, x_goal, Ts, dynamics, Q, Qf, R, x0, u_traj, N = acrobot_data()
    elif dyn_sys == "Cartpole":
        n, m, x_goal, Ts, dynamics, Q, Qf, R, x0, u_traj, N = cartpole_data()
    else:
        print("Dynamical system unknown. Please choose among 'Acrobot' or 'Cartpole'.")
        exit()

    model = Model(n, m, x_goal, Ts, dynamics, Q, Qf, R)
    ddp_controller = DDP(*model.return_ddp_args())
    x_traj, u_traj, J = ddp_controller.run_DDP(x0, u_traj, N)

    print('-'*40)
    print("* iLQR/DDP Controller *")
    print("Total cost: J = {}".format(J))
    print("Final state error: {}".format(jnp.linalg.norm(x_traj[:, -1] - x_goal)))
    print('-'*40)

    plot_trajectories(x_traj, u_traj)

    # Create animation
    if dyn_sys == "Acrobot":
        Plotter = AcrobotPlotter
    elif dyn_sys == "Cartpole":
        Plotter = CartpolePlotter

    plotter = Plotter()
    plotter.plot_animation(x_traj, Ts)


if __name__ == "__main__":
    main()

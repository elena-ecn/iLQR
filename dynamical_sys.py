import jax.numpy as jnp
from jax import jit


@jit
def cartpole_dynamics(x, u):
    """Computes the continuous-time dynamics for a cartpole ẋ=f(x,u).

    State is x = [p, θ, ṗ, θ̇]
    where p is the horizontal position and θ is the angle.
    θ = 0: pole hanging down, θ = 180: pole is up
    Inputs:
      - x(np.ndarray): The system state  [4x1]
      - u(np.ndarray): The control input [1x1]
    Returns:
      - x_d(np.ndarray): The time derivative of the state [4x1]
    """
    # Cartpole physical parameters
    params = {'mc': 1.0, 'mp': 0.2, 'l': 0.5}  # Cartpole parameters
    mc, mp, l = params['mc'], params['mp'], params['l']
    g = 9.81

    # State variables
    q = x[0:2].reshape(2, 1)
    q_d = x[2:4].reshape(2, 1)

    s = jnp.sin(q[1])[0]
    c = jnp.cos(q[1])[0]

    H = jnp.array([[mc+mp, mp*l*c], [mp*l*c, mp*l**2]])
    C = jnp.array([[0, -mp*q_d[1][0]*l*s], [0, 0]])
    G = jnp.array([[0], [mp*g*l*s]])
    B = jnp.array([[1], [0]])

    q_dd, _, _, _ = jnp.linalg.lstsq(-H, C@q_d + G - B*u[0], rcond=None)

    return jnp.vstack((q_d, q_dd)).reshape(4,)


@jit
def acrobot_dynamics(x, u):
    """Computes the continuous-time dynamics for an acrobot ẋ=f(x).

    A double-pendulum with actuation only at the elbow joint.
    Inputs:
      - x(np.ndarray): The system state  [4x1]
      - u(np.ndarray): The control input [1x1]
    Returns:
      - x_d(np.ndarray): The time derivative of the state [4x1]
    """
    # System parameters
    m1 = 1.0    # Mass of pendulum 1 [kg]
    m2 = 1.0    # Mass of pendulum 2 [kg]
    L1 = 1.0    # Length of pendulum 1 [m]
    L2 = 1.0    # Length of pendulum 2 [m]
    J1 = 1.0    # Link inertia
    J2 = 1.0    # Link inertia
    g = 9.8     # Gravitational acceleration [m/s^2]

    # State variables
    q1, q2, q1_d, q2_d = x

    c1 = jnp.cos(q1)
    s2 = jnp.sin(q2)
    c2 = jnp.cos(q2)
    c12 = jnp.cos(q1+q2)

    # Mass matrix
    m11 = m1*L1**2 + J1 + m2*(L1**2 + L2**2 + 2*L1*L2*c2) + J2
    m12 = m2*(L2**2 + L1*L2*c2 + J2)
    m22 = L2**2*m2 + J2
    M = jnp.block([[m11, m12],
                   [m12, m22]])

    # Bias term
    tmp = L1*L2*m2*s2
    b1 = -(2*q1_d*q2_d + q2_d**2)*tmp
    b2 = tmp*q1_d**2
    B = jnp.vstack((b1, b2))

    # Friction
    c = 1.0
    C = jnp.vstack((c*q1_d, c*q2_d))

    # Gravity term
    g1 = ((m1 + m2)*L2*c1 + m2*L2*c12)*g
    g2 = m2*L2*c12*g
    G = jnp.vstack((g1, g2))

    # Equations of motion
    tau = jnp.vstack((0.0, u[0]))
    q_dd, _, _, _ = jnp.linalg.lstsq(M, tau - B - G - C, rcond=None)

    x_d = jnp.vstack((q1_d, q2_d, q_dd)).reshape(4,)
    return x_d

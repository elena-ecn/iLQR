import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle


class CartpolePlotter:
    """Cartpole Animation"""
    def __init__(self):
        self.dt = None
        self.N = None

        # Cartpole parameters
        self.mc = 1.2  # Mass of cart       [kg]
        self.mp = 0.16 # Mass of pendulum   [kg]
        self.l = 0.55  # Length of pendulum [m]

        self.x1, self.y1 = None, None
        self.x2, self.y2 = None, None
        self.link = None
        self.mass = None
        self.cart = None
        self.time_string = None
        self.trace_steps = 20
        self.time_template = 'Time = %.1f s'
        self.cart_width = 0.5
        self.cart_height = 0.2

    def set_trajectories(self, x_history):
        """Computes the cartesian trajectories from the joint states."""
        self.N = x_history.shape[1]

        # Joint trajectories
        p = x_history.flatten()[:self.N]
        theta = x_history.flatten()[self.N:2*self.N]

        # Cartesian trajectories
        self.x1 = p                               # Cart position & pendulum origin
        self.y1 = 0*p
        self.x2 = self.x1 + self.l*np.sin(theta)  # Pendulum end
        self.y2 = self.y1 - self.l*np.cos(theta)

    def init(self):
        self.link.set_data([], [])
        self.mass.set_data([], [])
        self.time_string.set_text('')
        return self.link, self.mass, self.time_string

    def animate(self, i):
        """Draws each frame of the animation."""
        self.link.set_data([self.x1[i], self.x2[i]], [self.y1[i], self.y2[i]])
        self.mass.set_data([self.x2[i]], [self.y2[i]])
        self.time_string.set_text(self.time_template % (i*self.dt))
        self.ax.patches.remove(self.cart)
        x, y, w, h = self.x1[i]-self.cart_width/2, self.y1[i]-self.cart_height/2, self.cart_width, self.cart_height
        self.cart = Rectangle((x, y), w, h, color='k')
        self.ax.add_patch(self.cart)

        return self.link, self.mass, self.time_string

    def plot_animation(self, x_hist, dt):
        """Plots the double pendulum animation."""
        self.dt = dt
        self.set_trajectories(x_hist)

        sns.set_theme()
        fig, ax = plt.subplots()
        plt.title('Cartpole')
        plt.grid(True)
        plt.axis('square')
        plt.xlim([-3.0, 3.0])
        plt.ylim([-1.5, 1.5])
        ax.yaxis.set_ticklabels([])  # Hide axis tick labels

        self.time_string = ax.text(0.1, 0.91, '', transform=ax.transAxes)
        self.link, = ax.plot([], [], color='b', linestyle='-', linewidth=2)
        self.mass, = ax.plot([], [], color='b', marker='o', markersize=10)

        self.ax = ax
        x, y, w, h = self.x1[0]-self.cart_width/2, self.y1[0]-self.cart_height/2, self.cart_width, self.cart_height
        self.cart = Rectangle((x, y), w, h, color='k')
        self.ax.add_patch(self.cart)

        ax.hlines(0,-5,5, colors='black')  # Ground

        anim = FuncAnimation(fig, self.animate, init_func=self.init, frames=self.N, interval=1000*self.dt, blit=True)
        plt.show()

        # Save animation as gif
        anim.save('images/cartpole_animation.gif', writer=PillowWriter(fps=1/self.dt))


class AcrobotPlotter:
    """Acrobot Animation"""
    def __init__(self):
        self.dt = None
        self.N = None

        # Acrobot parameters
        self.m1 = 1.0  # Mass of pendulum 1 [kg]
        self.m2 = 1.0  # Mass of pendulum 2 [kg]
        self.L1 = 1.0  # Length of pendulum 1 [m]
        self.L2 = 1.0  # Length of pendulum 2 [m]
        self.g = 9.8   # Gravitational acceleration [m/s^2]

        self.x1, self.y1 = None, None
        self.x2, self.y2 = None, None
        self.link1, self.link2 = None, None
        self.mass1, self.mass2, self.m2_trace = None, None, None
        self.time_string = None
        self.trace_steps = 20
        self.time_template = 'Time = %.1f s'

    def set_trajectories(self, x_history):
        """Computes the cartesian trajectories from the joint states."""
        self.N = x_history.shape[1]

        # Joint trajectories
        q1 = x_history.flatten()[:self.N]
        q2 = x_history.flatten()[self.N:2*self.N]

        # Cartesian trajectories
        self.x1 = self.L1*np.cos(q1)                # First link endpoint
        self.y1 = self.L1*np.sin(q1)
        self.x2 = self.x1 + self.L2*np.cos(q1+q2)   # Second link endpoint
        self.y2 = self.y1 + self.L2*np.sin(q1+q2)

    def init(self):
        self.link1.set_data([], [])
        self.link2.set_data([], [])
        self.mass1.set_data([], [])
        self.mass2.set_data([], [])
        self.m2_trace.set_data([], [])
        self.time_string.set_text('')
        return self.link1, self.link2, self.mass1, self.mass2, self.m2_trace, self.time_string

    def animate(self, i):
        """Draws each frame of the animation."""
        self.link1.set_data([0, self.x1[i]], [0, self.y1[i]])
        self.link2.set_data([self.x1[i], self.x2[i]], [self.y1[i], self.y2[i]])
        self.mass1.set_data([self.x1[i]], [self.y1[i]])
        self.mass2.set_data([self.x2[i]], [self.y2[i]])
        self.m2_trace.set_data([self.x2[i-self.trace_steps: i]], [self.y2[i-self.trace_steps: i]])
        self.time_string.set_text(self.time_template % (i*self.dt))
        return self.link1, self.link2, self.mass1, self.mass2, self.m2_trace, self.time_string

    def plot_animation(self, x_hist, dt):
        """Plots the acrobot animation."""
        self.dt = dt
        self.set_trajectories(x_hist)

        sns.set_theme()
        fig, ax = plt.subplots()
        plt.title('Acrobot')
        plt.grid(True)
        plt.axis('square')
        plt.xlim([-self.L1-self.L2-0.5, self.L1+self.L2+0.5])
        plt.ylim([-self.L1-self.L2-0.5, self.L1+self.L2+0.5])
        ax.xaxis.set_ticklabels([])  # Hide axis tick labels
        ax.yaxis.set_ticklabels([])

        self.time_string = ax.text(0.1, 0.91, '', transform=ax.transAxes)
        self.link1, = ax.plot([], [], color='b', linestyle='-', linewidth=2)
        self.link2, = ax.plot([], [], color='b', linestyle='-', linewidth=2)
        self.mass1, = ax.plot([], [], color='b', marker='o', markersize=10)
        self.mass2, = ax.plot([], [], color='b', marker='o', markersize=10)
        self.m2_trace, = ax.plot([], [], 'r', marker='.', markersize=1, alpha=0.2, zorder=0)

        ax.plot(0, 0, marker='o', color='k', markersize=10, zorder=10)  # Base

        anim = FuncAnimation(fig, self.animate, init_func=self.init, frames=self.N, interval=1000*self.dt, blit=True)
        plt.show()

        # Save animation as gif
        anim.save('images/acrobot_animation.gif', writer=PillowWriter(fps=1000/self.dt))


def plot_trajectories(x_history, u_history):
    """Plots state & control trajectories."""

    n = x_history.shape[0]
    m = u_history.shape[0]

    # Plot state trajectories
    sns.set_theme()
    plt.figure()
    for i in range(n):
        plt.plot(x_history[i, :], label="x{}".format(i+1))
    plt.xlabel("N")
    plt.legend()
    plt.title("State trajectories")
    plt.savefig('images/state_trajectories.png')
    plt.show()

    # Plot control trajectories
    plt.figure()
    if m > 1:
        for i in range(m):
            plt.plot(u_history[i, :], label="u{}".format(i+1))
    else:
        plt.plot(u_history[0, :], label="u")
    plt.xlabel("N")
    plt.legend()
    plt.title("Control trajectories")
    plt.savefig('images/control_trajectories.png')
    plt.show()

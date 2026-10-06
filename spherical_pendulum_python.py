"""Driven spherical pendulum translated from a Mathematica Demonstration.

Dependencies:
    numpy
    scipy
    matplotlib

The nonlinear equations of motion are a direct translation of the two
Mathematica equations solved with NDSolve.  The original Manipulate interface
is represented here by editable parameters at the bottom of the file and by
functions for a phase plot and a 3-D animation.
"""

from dataclasses import dataclass, replace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp


@dataclass
class Parameters:
    # Physical/system parameters
    L: float = 1.0          # rod length
    mb: float = 20.0        # bob mass
    mr: float = 0.05        # rod mass
    mf: float = 0.30        # frame mass
    f: float = 0.50         # half-distance between frame rods
    tau: float = 0.0        # applied driving torque

    # Initial conditions
    theta0: float = np.deg2rad(-15.0)
    theta_dot0: float = 3.16
    phi0: float = 0.0
    phi_dot0: float = -0.35

    # Numerical / visualization parameters
    t_max: float = 30.0
    g: float = 9.81
    density: float = 7.9    # used only for visual rod/bob size, as in notebook


# Mathematica bookmarks translated to Python presets.
PRESETS = {
    "default": Parameters(),
    "full-swing pendulum": Parameters(
        L=1.0, mb=20.01, mf=0.2, mr=0.1,
        theta0=np.deg2rad(150), theta_dot0=1.64,
        phi_dot0=0.01,
    ),
    "almost there": Parameters(
        L=1.74, mb=25.0, mf=0.22, mr=0.14,
        theta0=np.deg2rad(178), theta_dot0=0.08,
        phi_dot0=-0.75,
    ),
    "3-star": Parameters(
        L=1.39, mb=20.21, mf=0.3, mr=0.05,
        theta0=np.deg2rad(45), theta_dot0=0.0,
        phi_dot0=0.08,
    ),
    "5-star": Parameters(
        L=1.75, mb=20.44, mf=0.3, mr=0.05,
        theta0=np.deg2rad(47), theta_dot0=0.0,
        phi_dot0=0.08,
    ),
    "13-star": Parameters(
        L=1.39, mb=19.84, mf=0.3, mr=0.05,
        theta0=np.deg2rad(84), theta_dot0=0.0,
        phi_dot0=0.08,
    ),
    "constant theta": Parameters(
        L=1.74, mb=15.51, mf=0.43, mr=0.01,
        theta0=np.deg2rad(50), theta_dot0=0.0,
        phi_dot0=2.96,
    ),
    "phi only": Parameters(
        L=1.74, mb=25.0, mf=0.5, mr=0.2,
        theta0=np.deg2rad(-6), theta_dot0=0.0,
        phi_dot0=-10.0,
    ),
    "torque only": Parameters(
        L=1.74, mb=25.0, mf=0.5, mr=0.2, tau=0.19,
        theta0=np.deg2rad(-2), theta_dot0=0.0,
        phi_dot0=0.0,
    ),
}


def equations_of_motion(t, y, p: Parameters):
    """First-order form of the Mathematica equations.

    State vector:
        y = [theta, theta_dot, phi, phi_dot]
    """
    theta, theta_dot, phi, phi_dot = y

    A = 3.0 * p.mb + p.mr
    B = 2.0 * p.mb + p.mr

    # From Mathematica equation 1:
    # -(1/6)L[-3g(2mb+mr)sin(theta)
    #          + L(3mb+mr)sin(2theta)phi_dot^2
    #          - 2L(3mb+mr)theta_ddot] == 0
    theta_ddot = (
        -3.0 * p.g * B * np.sin(theta)
        + p.L * A * np.sin(2.0 * theta) * phi_dot**2
    ) / (2.0 * p.L * A)

    # From Mathematica equation 2:
    # 3 tau == L^2(3mb+mr)sin(2theta)theta_dot phi_dot
    #          + [6 f^2 mf + L^2(3mb+mr)sin^2(theta)] phi_ddot
    denominator = (
        6.0 * p.f**2 * p.mf
        + p.L**2 * A * np.sin(theta)**2
    )

    phi_ddot = (
        3.0 * p.tau
        - p.L**2 * A * np.sin(2.0 * theta) * theta_dot * phi_dot
    ) / denominator

    return [theta_dot, theta_ddot, phi_dot, phi_ddot]


def solve_system(p: Parameters, samples=3001):
    """Numerically solve the coupled ODEs, equivalent to Mathematica NDSolve."""
    t_eval = np.linspace(0.0, p.t_max, samples)
    y0 = [p.theta0, p.theta_dot0, p.phi0, p.phi_dot0]

    sol = solve_ivp(
        equations_of_motion,
        (0.0, p.t_max),
        y0,
        args=(p,),
        t_eval=t_eval,
        dense_output=True,
        rtol=1e-9,
        atol=1e-11,
        max_step=0.02,
    )

    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def bob_radius(p: Parameters):
    """Same radius expression used in the Mathematica graphics."""
    return (3.0 * p.mb / (4.0 * np.pi * p.density)) ** (1.0 / 3.0)


def pendulum_xyz(theta, phi, length):
    """Position after x-rotation by theta and z-rotation by phi.

    This matches the trace coordinates used in the Mathematica notebook:
        {-sin(theta) sin(phi), sin(theta) cos(phi), -cos(theta)} * length
    """
    x = -length * np.sin(theta) * np.sin(phi)
    y =  length * np.sin(theta) * np.cos(phi)
    z = -length * np.cos(theta)
    return x, y, z


def rotate_z(x, y, phi):
    """Rotate xy coordinates by phi about the z-axis."""
    c, s = np.cos(phi), np.sin(phi)
    return c * x - s * y, s * x + c * y


def plot_phase(sol, p: Parameters, current_time=None):
    """Mathematica display=1: theta versus theta_dot phase curve."""
    theta = sol.y[0]
    theta_dot = sol.y[1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(theta, theta_dot, lw=1.8)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel(r"$\theta(t)$")
    ax.set_ylabel(r"$\dot{\theta}(t)$")
    ax.set_xticks(np.arange(-np.pi, np.pi + 0.01, np.pi / 2))
    ax.set_yticks(np.arange(-4, 5, 1))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Driven spherical pendulum: phase curve")

    if current_time is not None:
        current_time = np.clip(current_time, 0, p.t_max)
        theta_t, theta_dot_t, _, _ = sol.sol(current_time)
        ax.scatter([theta_t], [theta_dot_t], s=65, zorder=5)

    fig.tight_layout()
    return fig, ax


def animate_pendulum(sol, p: Parameters, trace=True, interval_ms=20):
    """Mathematica display=2: simplified 3-D pendulum + rotating frame animation."""
    t = sol.t
    theta = sol.y[0]
    phi = sol.y[2]

    # Use fewer frames for responsive plotting while keeping the numerical solution dense.
    frame_ids = np.linspace(0, len(t) - 1, min(900, len(t))).astype(int)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-2.37, 2.25)
    ax.set_box_aspect((4, 5, 4.62))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Driven spherical pendulum")

    # Table / base plane.
    xx, yy = np.meshgrid(np.linspace(-2, 2, 2), np.linspace(-2, 2, 2))
    zz = np.full_like(xx, -2.34)
    ax.plot_surface(xx, yy, zz, alpha=0.15)

    # Artists updated each frame.
    pendulum_line, = ax.plot([], [], [], lw=3)
    bob_point, = ax.plot([], [], [], marker="o", markersize=14, linestyle="None")
    trace_line, = ax.plot([], [], [], lw=1.2, alpha=0.65)

    frame_left, = ax.plot([], [], [], lw=4)
    frame_right, = ax.plot([], [], [], lw=4)
    frame_top, = ax.plot([], [], [], lw=4)
    frame_bottom, = ax.plot([], [], [], lw=4)
    suspension, = ax.plot([], [], [], lw=3)

    time_text = ax.text2D(0.03, 0.95, "", transform=ax.transAxes)

    # Trace uses the outer edge of the bob, as in the Mathematica ParametricPlot3D.
    trace_length = p.L + bob_radius(p)
    tx, ty, tz = pendulum_xyz(theta, phi, trace_length)

    def frame_segment(x_local, y_local, z_values, phi_now):
        x_rot, y_rot = rotate_z(np.asarray(x_local), np.asarray(y_local), phi_now)
        return x_rot, y_rot, np.asarray(z_values)

    def update(frame_number):
        i = frame_ids[frame_number]
        th = theta[i]
        ph = phi[i]

        # Pendulum center line.
        bx, by, bz = pendulum_xyz(th, ph, p.L)
        pendulum_line.set_data_3d([0, bx], [0, by], [0, bz])
        bob_point.set_data_3d([bx], [by], [bz])

        if trace:
            trace_line.set_data_3d(tx[:i + 1], ty[:i + 1], tz[:i + 1])
        else:
            trace_line.set_data_3d([], [], [])

        # Approximate the Mathematica rotating frame with rods/connector bars.
        x, y, z = frame_segment([-p.f, -p.f], [0, 0], [-2, 2], ph)
        frame_left.set_data_3d(x, y, z)
        x, y, z = frame_segment([p.f, p.f], [0, 0], [-2, 2], ph)
        frame_right.set_data_3d(x, y, z)
        x, y, z = frame_segment([-p.f, p.f], [0, 0], [2, 2], ph)
        frame_top.set_data_3d(x, y, z)
        x, y, z = frame_segment([-p.f, p.f], [0, 0], [-2, -2], ph)
        frame_bottom.set_data_3d(x, y, z)
        x, y, z = frame_segment([-p.f, p.f], [0, 0], [0, 0], ph)
        suspension.set_data_3d(x, y, z)

        time_text.set_text(f"elapsed time = {t[i]:.1f} s")

        return (
            pendulum_line, bob_point, trace_line,
            frame_left, frame_right, frame_top, frame_bottom, suspension,
            time_text,
        )

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    return fig, ani


def run(p=None, display="animation", trace=True):
    """Solve and show either the 3-D animation or the phase curve."""
    if p is None:
        p = PRESETS["default"]

    sol = solve_system(p)

    if display.lower() in {"phase", "phase curve", "1"}:
        plot_phase(sol, p)
        plt.show()
        return sol

    if display.lower() in {"animation", "pendulum", "2"}:
        fig, ani = animate_pendulum(sol, p, trace=trace)
        plt.show()
        return sol, ani

    raise ValueError("display must be 'animation'/'pendulum' or 'phase'/'phase curve'")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Edit this block as the Python equivalent of Mathematica Manipulate.
    # ------------------------------------------------------------------
    preset_name = "default"
    display = "animation"      # use "phase" for the phase curve
    trace = True

    p = PRESETS[preset_name]

    # Example of overriding individual parameters:
    # p = replace(
    #     p,
    #     L=1.20,
    #     mb=20.0,
    #     tau=0.10,
    #     theta0=np.deg2rad(30),
    #     theta_dot0=0.0,
    #     phi_dot0=1.0,
    # )

    run(p, display=display, trace=trace)

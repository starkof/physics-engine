import math
import scipy.constants as constants
import matplotlib.pyplot as plt
import numpy as np

# todo: use numpy for array calculations


def potential_energy(h, m):
    """

    :param h: height in meters
    :param m: mass in kilograms
    :return: potential energy measured in joules
    """
    return m * h * constants.g


def kinetic_energy(m, v):
    """

    :param m: mass in kilograms
    :param v: velocity in meters per second
    :return: kinetic energy in joules
    """
    return 0.5 * m * v ** 2


def distance(v_0, d_0, a, t):
    """

    :param v_0: initial velocity (m/s)
    :param d_0: distance (m)
    :param a: acceleration (m/s^2)
    :param t: time in seconds (s)
    :return: distance travelled in meters
    """
    return v_0*t + 0.5*a*t**2 + d_0


def velocity(v_0, a, t):
    """

    :param v_0: initial velocity (m/s)
    :param a: acceleration (m/s^2)
    :param t: time in seconds (s)
    :return: velocity (m/s)
    """
    return v_0 + a*t


def final_velocity(v_0, a, d):
    """

    :param v_0:
    :param a:
    :param d:
    :return:
    """
    return v_0**2 + 2*a*d


def time_to_fall(v_0, h):
    """

    :param v_0: velocity (m/s)
    :param h: height (m)
    :return: time (s)
    """
    v_f = final_velocity(v_0, constants.g, h)
    return (2*h)/(v_0 + v_f)


def simulate(time_step, total_time, initial_velocity, initial_distance, acceleration):
    t = np.linspace(0, total_time, int(total_time/time_step))

    d = distance(initial_velocity, initial_distance, acceleration, t)
    v = velocity(initial_velocity, acceleration, t)

    fig, axs = plt.subplots(2)
    axs[0].plot(t, d)
    axs[1].plot(t, v)
    plt.show()


if __name__ == '__main__':
    simulate(0.1, 10, 0, 0, constants.g)

import math
import scipy.constants as constants
import matplotlib.pyplot as plt

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


def velocity_with_acceleration(v_0, a, t):
    """

    :param v_0: initial velocity (m/s)
    :param a: acceleration (m/s^2)
    :param t: time in seconds (s)
    :return: velocity (m/s)
    """
    return v_0 + a*t


def velocity_with_distance_and_acceleration(v_0, a, d):
    """

    :param v_0: initial velocity
    :param a: acceleration
    :param d: distance
    :return:
    """
    return math.sqrt(v_0**2 + 2*a*d)


def velocity_of_falling_object(initial_height, initial_speed):
    # todo: calculate elapsed time for each step
    # todo: the equation used here doesn't account for negative velocities
    steps = 100
    step_distance = initial_height/steps
    velocities = []

    # print(initial_speed)
    velocities.append(initial_speed)
    for i in range(steps):
        initial_speed = velocity_with_distance_and_acceleration(initial_speed, constants.g, step_distance)
        print(initial_speed)
        velocities.append(initial_speed)

    return velocities


v = velocity_of_falling_object(1000, -100)

plt.plot(v)
plt.show()


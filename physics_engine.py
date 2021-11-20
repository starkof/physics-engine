import scipy.constants as constants
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from typing import List


def potential_energy(h, m):
    """

    :param h: height in meters
    :param m: mass in kilograms
    :return: potential energy measured in joules
    """
    return m * constants.g * h


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


def plot_velocity_and_distance(t, v, d):
    fig, axs = plt.subplots(2)
    axs[0].plot(t, d)
    axs[1].plot(t, v)
    plt.show()


def plot_energy(ke, pe):
    plt.plot(ke)
    plt.plot(pe)
    plt.show()


def plot_3d(x, y, z):
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z)
    plt.show()


@dataclass
class PointVector1D:
    velocity: float
    acceleration: float
    point: float = 0


@dataclass
class PointVectorGroup1D:
    point: float
    vectors: List[PointVector1D]

    resolved_vector: PointVector1D = PointVector1D(0, 0, 0)

    def resolve(self):
        for v in self.vectors:
            self.resolved_vector.velocity += v.velocity
            self.resolved_vector.acceleration += v.acceleration
        self.resolved_vector.point = self.point

    def get_resolved(self):
        return self.resolved_vector


def simulate_1d(time_step, total_time, vectors: List[PointVectorGroup1D]):
    t = np.linspace(0, total_time, int(total_time / time_step))
    d_n = np.zeros(t.size)
    v_n = np.zeros(t.size)

    for vec in vectors:
        vec.resolve()
        resolved = vec.get_resolved()

        d_n = np.row_stack((
            d_n,
            distance(resolved.velocity, resolved.point, resolved.acceleration, t)
        ))
        v_n = np.row_stack((
            v_n,
            velocity(resolved.velocity, resolved.acceleration, t)
        ))

    assert v_n.shape == d_n.shape, 'Velocity and distance matrices must have the same shape'

    for i in range(1, d_n.shape[0]):
        plot_velocity_and_distance(t, v_n[i], d_n[i])


# todo: consider having a common parent for Vector2D and Point2D
@dataclass
class Vector2D:
    x: float
    y: float

    def __str__(self):
        return '[{}\n{}]'.format(self.x, self.y)

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y

        return Vector2D(x, y)


@dataclass
class Point2D:
    x: float
    y: float

    def __str__(self):
        return '[{}\n{}]'.format(self.x, self.y)

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y

        return Vector2D(x, y)


@dataclass
class PointVector2D:
    velocity: Vector2D = Vector2D(0, 0)
    acceleration: Vector2D = Vector2D(0, 0)
    point: Point2D = Point2D(0, 0)


@dataclass
class PointVectorGroup2D:
    point: Point2D
    vectors: List[PointVector2D]

    resolved_vector: PointVector2D()

    def resolve(self):
        for v in self.vectors:
            self.resolved_vector.velocity += v.velocity
            self.resolved_vector.acceleration += v.acceleration
        self.resolved_vector.point = self.point

    def get_resolved(self):
        return self.resolved_vector


def simulate_2d(time_step, total_time, vectors: List[PointVector2D]):
    t = np.linspace(0, total_time, int(total_time / time_step))
    pass


def main():
    # simulate_1d(0.1, 10, [
    #     PointVectorGroup1D(0, [
    #         PointVector1D(0, constants.g),
    #         PointVector1D(-100, 20)
    #     ]),
    #     PointVectorGroup1D(1000, [
    #         PointVector1D(0, constants.g),
    #         PointVector1D(0, 20),
    #         PointVector1D(0, 10)
    #     ])
    # ])
    pass


if __name__ == '__main__':
    # v = PointVectorGroup2D()
    pass

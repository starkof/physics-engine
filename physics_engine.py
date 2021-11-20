import scipy.constants as constants
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from dataclasses import dataclass, field
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
    # print(v_0*t)
    # print(0.5*a*t**2)
    # print(d_0)
    # print()
    # print(v_0*t + 0.5*a*t**2)
    # print(v_0*t + 0.5*a*t**2 + d_0)

    # d_0 must come first to ensure correct output with numpy
    return d_0 + v_0*t + 0.5*a*t**2


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


@dataclass
class Array2D:
    x: float
    y: float

    def _to_np(self):
        return np.array([[self.x], [self.y]])

    def __str__(self):
        return '[[{}]\n [{}]]'.format(self.x, self.y)

    def __add__(self, other):
        if type(other) == Array2D:
            return self._to_np() + other._to_np()

        return self._to_np() + other

    def __radd__(self, other):
        if type(other) == Array2D:
            return self._to_np() + other._to_np()

        return self._to_np() + other
        # return self.__add__(other)

    def __pos__(self):
        return self._to_np()

    def __sub__(self, other):
        if type(other) == Array2D:
            return self._to_np() - other._to_np()
        return self._to_np() - other

    def __rsub__(self, other):
        if type(other) == Array2D:
            return other._to_np() - self._to_np()

        return other - self._to_np()

    def __neg__(self):
        return -1 * self._to_np()

    def __mul__(self, other):
        if type(other) == Array2D:
            return other._to_np() * self._to_np()

        return self._to_np() * other

    def __rmul__(self, other):
        if type(other) == Array2D:
            return other._to_np() * self._to_np()

        return other * self._to_np()

    def __truediv__(self, other):
        if type(other) == Array2D:
            return self._to_np() / other._to_np()

        return self._to_np() / other

    def __rtruediv__(self, other):
        if type(other) == Array2D:
            return other._to_np() / self._to_np()

        return other / self._to_np()

    def __floordiv__(self, other):
        if type(other) == Array2D:
            return self._to_np() // other._to_np()

        return self._to_np() // other

    def __rfloordiv__(self, other):
        if type(other) == Array2D:
            return other._to_np() // self._to_np()

        return other // self._to_np()

    def __mod__(self, other):
        if type(other) == Array2D:
            return other._to_np() % self._to_np()
        return self._to_np() % other

    def __rmod__(self, other):
        if type(other) == Array2D:
            return other._to_np() % self._to_np()

        return other % self._to_np()

    def __pow__(self, power, modulo=None):
        return self._to_np() ** power


@dataclass
class PointVector2D:
    velocity: Array2D
    acceleration: Array2D
    point: Array2D = Array2D(0, 0)


@dataclass
class PointVectorGroup2D:
    point: Array2D
    vectors: List[PointVector2D]

    resolved_vector: PointVector2D = PointVector2D(Array2D(0, 0), Array2D(0, 0))

    def resolve(self):
        for v in self.vectors:
            self.resolved_vector.velocity += v.velocity
            self.resolved_vector.acceleration += v.acceleration
        self.resolved_vector.point = self.point

    def get_resolved(self):
        return self.resolved_vector


@dataclass
class SimulationResult:
    time: npt.NDArray = np.array([])
    positions: List[npt.NDArray] = field(default_factory=list)
    velocities: List[npt.NDArray] = field(default_factory=list)

    def append_velocity(self, velocity_matrix: npt.NDArray):
        self.velocities.append(velocity_matrix)

    def append_position(self, position_matrix: npt.NDArray):
        self.positions.append(position_matrix)

    def set_time(self, time_array: npt.NDArray):
        self.time = time_array


def simulate_2d(time_step, total_time, vectors: List[PointVectorGroup2D]):
    t = np.linspace(0, total_time, int(total_time / time_step))
    result = SimulationResult()

    for vec in vectors:
        vec.resolve()
        resolved = vec.get_resolved()
        v = velocity(resolved.velocity, resolved.acceleration, t)
        d = distance(resolved.velocity, resolved.point, resolved.acceleration, t)

        # print(resolved.velocity)
        # print(resolved.point)
        # print(resolved.acceleration)

        # print(v.shape)
        # print(d.shape)

        result.append_velocity(v)
        result.append_position(d)

    return result


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
    res = simulate_2d(0.1, 10, [
        PointVectorGroup2D(Array2D(0, 0), [
            PointVector2D(Array2D(5, 5), Array2D(5, 5))
        ])
    ])

    print(res.positions)
    print(res.velocities)
    # print(res.positions[0][0][0])
    # plot_3d()

    # ax = plt.axes(projection='3d')
    # ax.plot3D(res.time, res.positions)
    # plt.show()


if __name__ == '__main__':
    # # v = PointVectorGroup2D()
    # arr = Array2D(1, 2)
    # # print(arr.__to_np())
    # print(arr)
    main()

    # a = Array2D(1, 2)
    # arr = np.array([[1], [2]])
    # b = np.array([[1, 2, 3], [4, 5, 6]])
    # print(b)
    # print(a)
    # print((a + b)[0][0])
    # print(b + a)
    # print(type((b + a)[0][0][0]))
    # print(b + arr)
    # print(arr + b)
    # print(0.5 + a)
    # print(a + a)
    # print(a + 0.5)

    # print()
    # print(a - 1)
    # print(1 - a)
    # print(a - a)
    # print()
    # print(a**2)
    # print()
    # print(2 * a)
    # print(a * 2)
    # print(a * a)


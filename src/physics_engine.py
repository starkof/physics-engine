import scipy.constants as const
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from dataclasses import dataclass, field
from typing import List


def array(*args):
    assert len(args) > 0, 'Array must have one or more elements'
    return np.array([[x] for x in args])


def potential_energy(h, m):
    """

    :param h: height in meters
    :param m: mass in kilograms
    :return: potential energy measured in joules
    """
    return m * const.g * h


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
class PointVector:
    velocity: npt.NDArray
    acceleration: npt.NDArray
    point: npt.NDArray = array(0, 0, 0)


@dataclass
class PointVectorGroup:
    point: npt.NDArray
    vectors: List[PointVector]
    dimensions: int = 2

    __resolved_vector: PointVector = PointVector(array(0), array(0))

    def resolve(self):
        self.__resolved_vector = PointVector(array(*([0] * self.dimensions)), array(*([0] * self.dimensions)))
        for v in self.vectors:
            assert v.velocity.shape == (self.dimensions, 1) and v.acceleration.shape == (self.dimensions, 1), \
                '{0}D array must have {0} rows and 1 column'.format(self.dimensions)

            self.__resolved_vector.velocity += v.velocity
            self.__resolved_vector.acceleration += v.acceleration
        self.__resolved_vector.point = self.point

    def get_resolved(self):
        return self.__resolved_vector


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


def simulate(time_step, total_time, vectors: List[PointVectorGroup]) -> SimulationResult:
    t = np.linspace(0, total_time, int(total_time / time_step))
    result = SimulationResult()

    for vec in vectors:
        vec.resolve()
        resolved = vec.get_resolved()
        v = velocity(resolved.velocity, resolved.acceleration, t)
        d = distance(resolved.velocity, resolved.point, resolved.acceleration, t)

        result.append_velocity(v)
        result.append_position(d)

    return result


def main():
    dimensions = 3
    res = simulate(0.5, 10, [
        PointVectorGroup(array(0, 0, 0),
                         [
                             PointVector(array(5, 5, 5), array(5, 5, 5))
                         ],
                         dimensions),
        PointVectorGroup(array(0, 0, 0),
                         [
                             PointVector(array(5, 4, 3), array(2, 1, 0))
                         ],
                         dimensions)
    ])


if __name__ == '__main__':
    main()

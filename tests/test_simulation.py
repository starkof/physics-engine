import unittest

import numpy as np

from src.physics_engine import *


class TestSimulation(unittest.TestCase):
    def test_SimulationResult(self):
        result = SimulationResult(np.array([1, 2, 3]), [np.array([4, 5, 6])], [np.array([7, 8, 9])])
        self.assertTrue(
            np.array_equal(result.time, np.array([1, 2, 3]))
        )
        self.assertTrue(
            np.array_equal(result.positions[0], np.array([4, 5, 6]))
        )
        self.assertTrue(
            np.array_equal(result.velocities[0], np.array([7, 8, 9]))
        )

        result.append_position(np.array([8, 9, 10]))
        result.append_velocity(np.array([11, 12, 13]))
        self.assertTrue(
            np.array_equal(result.positions[1], np.array([8, 9, 10]))
        )
        self.assertTrue(
            np.array_equal(result.positions[1], np.array([8, 9, 10]))
        )

    def test_simulate(self):
        dimensions = 3
        res = simulate(6, 5, [
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

        expected_time = np.array([0, 1, 2, 3, 4, 5])
        expected_positions = [np.array([[0., 7.5, 20., 37.5, 60., 87.5],
                                        [0., 7.5, 20., 37.5, 60., 87.5],
                                        [0., 7.5, 20., 37.5, 60., 87.5]]),
                              np.array([[0., 6., 14., 24., 36., 50.],
                                        [0., 4.5, 10., 16.5, 24., 32.5],
                                        [0., 3., 6., 9., 12., 15.]])]
        expected_velocities = [np.array([[5., 10., 15., 20., 25., 30.],
                                         [5., 10., 15., 20., 25., 30.],
                                         [5., 10., 15., 20., 25., 30.]]),
                               np.array([[5., 7., 9., 11., 13., 15.],
                                         [4., 5., 6., 7., 8., 9.],
                                         [3., 3., 3., 3., 3., 3.]])]

        self.assertTrue(np.array_equal(expected_time, res.time))
        self.assertTrue(np.array_equal(expected_positions[0], res.positions[0]))
        self.assertTrue(np.array_equal(expected_positions[1], res.positions[1]))

        self.assertTrue(np.array_equal(expected_velocities[0], res.velocities[0]))
        self.assertTrue(np.array_equal(expected_velocities[1], res.velocities[1]))

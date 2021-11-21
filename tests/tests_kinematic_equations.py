import unittest
from src.physics_engine import *
import scipy.constants as const


class TestEquations(unittest.TestCase):
    def test_velocity_scalar(self):
        self.assertEqual(0, velocity(0, 0, 0))
        self.assertEqual(0, velocity(0, 1, 0))
        self.assertEqual(1, velocity(0, 1, 1))
        self.assertEqual(5, velocity(5, 0, 0))
        self.assertEqual(11, velocity(5, 2, 3))

    def test_velocity_vector(self):
        raise NotImplemented

    def test_distance_scalar(self):
        self.assertEqual(0, distance(0, 0, 0, 0))
        self.assertEqual(0, distance(0, 0, 1, 0))
        self.assertEqual(0.5, distance(0, 0, 1, 1))
        self.assertEqual(2, distance(0, 0, 1, 2))
        self.assertEqual(5, distance(5, 0, 0, 1))
        self.assertEqual(6, distance(0, 6, 0, 0))
        self.assertEqual(16, distance(2, 6, 3, 2))

    def test_distance_vector(self):
        raise NotImplemented

    def test_potential_energy_scalar(self):
        self.assertEqual(0, potential_energy(0, 0))
        self.assertEqual(const.g, potential_energy(1, 1))
        self.assertEqual(58.8399, potential_energy(2, 3))

    def test_potential_energy_vector(self):
        raise NotImplemented

    def test_kinetic_energy_scalar(self):
        self.assertEqual(0, kinetic_energy(0, 0))
        self.assertEqual(0.5, kinetic_energy(1, 1))
        self.assertEqual(2, kinetic_energy(1, 2))
        self.assertEqual(6, kinetic_energy(3, 2))

    def test_kinetic_energy_vector(self):
        raise NotImplemented

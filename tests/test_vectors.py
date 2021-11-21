import unittest
import numpy as np
from src.physics_engine import *


class TestVectors(unittest.TestCase):
    def test_array(self):
        self.assertTrue(
            np.array_equal(np.array([[1]]), array(1))
        )
        self.assertTrue(
            np.array_equal(np.array([[1], [2]]), array(1, 2))
        )
        self.assertTrue(
            np.array_equal(np.array([[1], [2], [3]]), array(1, 2, 3))
        )
        self.assertTrue(
            np.array_equal(np.array([[1], [2], [3], [4]]), array(1, 2, 3, 4))
        )
        with self.assertRaises(AssertionError):
            array()

    def test_PointVector(self):
        vector1 = PointVector(array(1), array(2))
        self.assertEqual(array(1), vector1.velocity)
        self.assertEqual(array(2), vector1.acceleration)

        vector2 = PointVector(array(1), array(2), array(3))
        self.assertEqual(array(1), vector2.velocity)
        self.assertEqual(array(2), vector2.acceleration)
        self.assertEqual(array(3), vector2.point)

    def test_PointVectorGroup_validation(self):
        vector_list_1d = [
            PointVector(array(4), array(6)),
            PointVector(array(8), array(9))
        ]

        vector = PointVectorGroup(array(1, 2), vector_list_1d)
        with self.assertRaises(AssertionError):
            vector.resolve()

    def test_PointVectorGroup_1D(self):
        vector_list = [
            PointVector(array(4), array(6)),
            PointVector(array(8), array(9))
        ]

        vector = PointVectorGroup(array(1), vector_list, 1)
        vector.resolve()
        self.assertTrue(
            np.array_equal(array(12), vector.get_resolved().velocity)
        )
        self.assertTrue(
            np.array_equal(array(15), vector.get_resolved().acceleration)
        )
        self.assertTrue(
            np.array_equal(array(1), vector.get_resolved().point)
        )

    def test_PointVectorGroup_2D(self):
        vector_list_2d = [
            PointVector(array(4, 5), array(6, 7)),
            PointVector(array(1, 2), array(3, 5))
        ]

        vector = PointVectorGroup(array(1, 2), vector_list_2d, 2)
        vector.resolve()
        self.assertTrue(
            np.array_equal(array(5, 7), vector.get_resolved().velocity)
        )
        self.assertTrue(
            np.array_equal(array(9, 12), vector.get_resolved().acceleration)
        )
        self.assertTrue(
            np.array_equal(array(1, 2), vector.get_resolved().point)
        )

    def test_PointVectorGroup_3D(self):
        vector_list = [
            PointVector(array(4, 5, 6), array(6, 7, 8)),
            PointVector(array(1, 2, 3), array(3, 5, 7))
        ]

        vector = PointVectorGroup(array(1, 2, 3), vector_list, 3)
        vector.resolve()
        self.assertTrue(
            np.array_equal(array(5, 7, 9), vector.get_resolved().velocity)
        )
        self.assertTrue(
            np.array_equal(array(9, 12, 15), vector.get_resolved().acceleration)
        )
        self.assertTrue(
            np.array_equal(array(1, 2, 3), vector.get_resolved().point)
        )

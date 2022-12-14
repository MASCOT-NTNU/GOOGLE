""" Unittest for T class. """

from unittest import TestCase
from Simulators.T import T


class TestT(TestCase):

    def setUp(self) -> None:
        self.t = T()

    def test_t(self) -> None:
        value = 10.2
        self.t.set_t_steering(value)
        self.assertEqual(self.t.get_t_steering(), value)



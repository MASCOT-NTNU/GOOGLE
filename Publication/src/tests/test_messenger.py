""" Unit test for Messenger
This module tests the Messenger object.
"""

from unittest import TestCase
from AUVSimulator.Messenger import Messenger


class TestMessenger(TestCase):

    def setUp(self) -> None:
        self.msg = Messenger()

    def test_send_sms(self):
        m = self.msg.send_sms()
        self.assertEqual(m, "SMS")

    def test_send_acoustic(self):
        m = self.msg.send_acoustic()
        self.assertEqual(m, "ACOUSTIC")

    def test_send_iridium(self):
        m = self.msg.send_iridium()
        self.assertEqual(m, "IRIDIUM")

    def test_send_4g(self):
        m = self.msg.send_4g()
        self.assertEqual(m, "4G")


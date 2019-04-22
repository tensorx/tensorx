from unittest import TestCase
from tensorx.callbacks import *
from tensorx.layers import Param


class TestCallbacks(TestCase):
    def test_OnEveryStep(self):
        t1 = OnStep(2)
        t2 = OnEveryStep(2)

        self.assertEqual(t1, t2)
        self.assertEqual(t2, t1)

        t1 = OnStep(10)
        t2 = OnEveryStep(2)

        self.assertEqual(t1, t2)
        self.assertEqual(t2, t1)

    def test_OnEveryEpochStep(self):
        t1 = OnStep(2)
        t2 = OnEveryEpochStep(2)
        t3 = OnEpochStep(4)

        self.assertNotEqual(t1, t2)
        self.assertNotEqual(t1, t3)
        self.assertEqual(t2, t3)

    def test_Param_Property(self):
        p1 = Param(value=3, name="test")

        self.assertIsInstance(p1, Param)
        self.assertNotIsInstance(p1, Property)

        class Obs:
            def trigger(self, event):
                print(event)

        obs = Obs()
        p1.register(obs)

        p1.value = 3


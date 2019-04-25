from unittest import TestCase
from tensorx.callbacks import *
from tensorx.layers import Param
from functools import partial


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

        assert_value_change = partial(self.assertIsInstance, cls=OnValueChange)

        class Obs:
            def trigger(self, event):
                assert_value_change(event)

        obs = Obs()
        p1.register(obs)
        p1.value = 3

    def test_event_hash(self):
        e1 = OnValueChange("a")
        e2 = OnValueChange("b")

        d = {e1: 2}
        self.assertIn(e1, d)
        self.assertNotIn(e2, d)

    def test_CSVWritter(self):
        prop_a = Property("a", 1)
        prop_b = Property("b", 1)
        prop_c = Property("c", 1)
        triggered = None

        scheduler = Scheduler(obj=None, properties=[prop_a, prop_b, prop_c])

        cb = Callback(OnValueChange("a"), fn=lambda *_: print("a"))
        scheduler.register(cb)

        prop_b.value = 2
        prop_a.value = 2

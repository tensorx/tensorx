import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import unittest
import tensorx as tx
from tensorx.train.callbacks import *
from functools import partial
from tensorx.layers import Param


class MyTestCase(unittest.TestCase):
    def test_param_trigger(self):
        p1 = tx.Param(init_value=3, name="test")
        self.assertEqual(p1.n_units, 0)
        # TODO should guarantee that shape always returns a TensorShape?
        #   using a property would get this done
        self.assertListEqual(p1.shape.as_list(), [])

        self.assertIsInstance(p1, tx.Param)
        self.assertNotIsInstance(p1, Property)

        assert_value_change = partial(self.assertIsInstance, cls=OnValueChange)

        class Obs:
            def __init__(self):
                self.obj = None

            def listen(self, obj):
                self.obj = obj
                obj.register(self)

            def trigger(self, event):
                assert_value_change(event)

        obs = Obs()
        obs.listen(p1)
        p1.value = 3
        self.assertEqual(p1().numpy(), p1.value.numpy())

        fn = p1.as_function()
        self.assertEqual(fn().numpy(), p1.value.numpy())

        # there should be no problem with compilation to tf graph since
        # triggers occur in set value only, and the graph works from a variable read
        p1.value = 4
        self.assertEqual(fn().numpy(), 4)

    def test_OnEveryStep(self):
        t1 = OnStep(2)
        t2 = OnEveryStep(2)
        t3 = OnEveryStep(1)
        t4 = OnEveryStep(3)
        t5 = OnStep(6, AT.END)
        t6 = OnStep(6, AT.START)

        self.assertNotEqual(t1, t2)
        self.assertTrue(t2.match(t1))
        self.assertTrue(t1.match(t2))
        self.assertTrue(t1.match(t3))
        self.assertTrue(t3.match(t1))

        self.assertTrue(t3.match(t4))
        self.assertFalse(t4.match(t3))

        self.assertTrue(t4.match(t5))
        self.assertTrue(t5.match(t4))
        self.assertFalse(t4.match(t6))

    def test_OnEpoch(self):
        t1 = OnEpoch(1)
        t2 = OnEpoch(2)
        t3 = OnEveryEpoch(1)

        self.assertFalse(t1.match(t2))
        self.assertTrue(t1.match(t3))
        self.assertTrue(t3.match(t1))
        self.assertTrue(t2.match(t3))
        self.assertTrue(t3.match(t2))

    def test_OnEveryEpochStep(self):
        t1 = OnStep(2)
        t2 = OnEveryEpochStep(2)
        t3 = OnEpochStep(4)
        t4 = OnStep(2)
        t5 = OnEpochStep(2)

        self.assertFalse(t1.match(t5))
        self.assertFalse(t5.match(t1))

        self.assertEqual(t1, t4)
        self.assertNotEqual(t1, t2)
        self.assertNotEqual(t1, t3)
        self.assertTrue(t2.match(t3))

    def test_Param_Property(self):
        p1 = Param(init_value=3, name="test")

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

        e3 = OnEpoch(1)
        e4 = OnEveryEpoch(1)
        d = {e3: 2, e4: 3}

        e5 = OnStep(2)
        e6 = OnEveryStep(2)
        d = {e5: 1, e6: 2}

        e7 = OnLoop(at=AT.START)
        d = {e7: 1}

    def test_scheduler(self):
        prop_a = Property("a", 1)
        prop_b = Property("b", 1)
        prop_a2 = Property("a2", 1)
        prop_b2 = Property("b2", 1)

        changed = []

        scheduler = Scheduler(model=None, properties=[prop_a, prop_b, prop_a2, prop_b2])

        def fna(*args):
            prop_a2.value = 2
            changed.append(1)

        def fnb(*args):
            prop_b2.value = 2
            changed.append(2)

        cb1 = Callback({OnValueChange("a"): fna}, priority=-1)
        cb2 = Callback({OnValueChange("a"): fnb}, priority=-2)
        # if this executed fnb it would change b2 value which would result in a recursive call
        cb3 = Callback({OnValueChange("b2"): fna})
        scheduler.register(cb1)
        scheduler.register(cb2)
        scheduler.register(cb3)

        prop_a.value = 2
        self.assertEqual(changed, [1, 2, 1])

    def test_on_callback(self):
        a = Property("a", 1)
        c = Property("b", 2)

        def fn1(*args):
            c.value -= 1

        def fn2(*args):
            c.value *= 2

        def fn3(*args):
            c.value *= 2

        cb1 = Callback({OnValueChange("a"): fn1}, priority=1)
        cb2 = Callback({OnCallback(cb1, at=AT.START): fn2}, priority=1)
        cb3 = Callback({OnCallback(cb1, at=AT.END): fn3}, priority=1)

        scheduler = Scheduler(model=None, properties=[a])
        scheduler.register(cb1)
        scheduler.register(cb2)
        scheduler.register(cb3)

        self.assertEqual(c.value, 2)
        a.value = 2
        self.assertEqual(c.value, 6)

        # scheduler.register(cb2)


if __name__ == '__main__':
    unittest.main()

import unittest
import tensorx as tx
import tensorx.callbacks as tc
from functools import partial
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class MyTestCase(unittest.TestCase):
    def test_param_trigger(self):
        p1 = tx.Param(value=3, name="test")
        self.assertEqual(p1.n_units, 0)
        # TODO should guarantee that shape always returns a TensorShape?
        #   using a property would get this done
        self.assertListEqual(p1.shape.as_list(), [])

        self.assertIsInstance(p1, tx.Param)
        self.assertNotIsInstance(p1, tc.Property)

        assert_value_change = partial(self.assertIsInstance, cls=tc.OnValueChange)

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

        fn = p1.compile_graph()
        self.assertEqual(fn().numpy(), p1.value.numpy())

        # there should be no problem with compilation to tf graph since
        # triggers occur in set value only, and the graph works from a variable read
        p1.value = 4
        self.assertEqual(fn().numpy(), 4)


if __name__ == '__main__':
    unittest.main()

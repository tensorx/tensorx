import unittest
from tensorflow.python.framework import ops
import contextlib
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.eager import context
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import device_lib
from tensorflow.python.util import compat
from tensorflow.python.ops.variables import Variable
import tensorflow as tf
import logging
import numpy as np


def gpu_device_name():
    """Returns the name of a GPU device if available or the empty string."""
    for x in device_lib.list_local_devices():
        if x.device_type == "GPU" or x.device_type == "SYCL":
            return compat.as_str(x.name)
    return ""


class ErrorLoggingSession(session.Session):
    """Wrapper around a Session that logs errors in run().
    """

    def run(self, *args, **kwargs):
        try:
            return super(ErrorLoggingSession, self).run(*args, **kwargs)
        except Exception as e:  # pylint: disable=broad-except
            # Note: disable the logging for OutOfRangeError, which makes the output
            # of tf.data tests hard to read, because OutOfRangeError is used as the
            # signal completion
            if not isinstance(e, errors.OutOfRangeError):
                logging.error(str(e))
            raise


class TestCase(unittest.TestCase):

    def __init__(self, method_name="run_tensorx_test"):
        super().__init__(method_name)
        self._cached_session = None

    def setUp(self):
        self._clear_cached_session()
        ops._default_graph_stack.reset()
        ops.reset_default_graph()

    def eval(self, tensors, feed_dict=None):
        """Evaluates tensors and returns numpy values.

        Args:
          tensors: A Tensor or a nested list/tuple of Tensors.

        Returns:
          tensors numpy values.
        """
        if context.executing_eagerly():
            return self._eval_helper(tensors)
        else:
            sess = ops.get_default_session()
            if sess is None:
                with self.cached_session() as sess:
                    return sess.run(tensors, feed_dict=feed_dict)
            else:
                return sess.run(tensors, feed_dict=feed_dict)

    def assertGreater(self, a, b, msg=None):
        if isinstance(a, (Tensor, Variable)):
            a = self.eval(a)
        if isinstance(b, (Tensor, Variable)):
            b = self.eval(b)

        super().assertGreater(a, b, msg)

    def assertEqual(self, first, second, msg=None):
        if isinstance(first, (Tensor, Variable)):
            first = self.eval(first)
        if isinstance(second, (Tensor, Variable)):
            second = self.eval(second)

        super().assertEqual(first, second)

    def assertShapeEqual(self, first, second):
        first = tf.convert_to_tensor(first)
        second = tf.convert_to_tensor(second)

        self.assertAllEqual(first.get_shape().as_list(), second.get_shape().as_list())

    def _GetNdArray(self, a):
        # If a is a tensor then convert it to ndarray
        if isinstance(a, ops.Tensor):
            if isinstance(a, ops._EagerTensorBase):
                a = a.numpy()
            else:
                a = self.evaluate(a)
        if not isinstance(a, np.ndarray):
            return np.array(a)

        return a

    def assertAllCloseAccordingToType(self,
                                      a,
                                      b,
                                      rtol=1e-6,
                                      atol=1e-6,
                                      float_rtol=1e-6,
                                      float_atol=1e-6,
                                      half_rtol=1e-3,
                                      half_atol=1e-3,
                                      bfloat16_rtol=1e-2,
                                      bfloat16_atol=1e-2,
                                      msg=None):
        """Like assertAllClose, but also suitable for comparing fp16 arrays.
        In particular, the tolerance is reduced to 1e-3 if at least
        one of the arguments is of type float16.
        Args:
          a: the expected numpy ndarray or anything can be converted to one.
          b: the actual numpy ndarray or anything can be converted to one.
          rtol: relative tolerance.
          atol: absolute tolerance.
          float_rtol: relative tolerance for float32.
          float_atol: absolute tolerance for float32.
          half_rtol: relative tolerance for float16.
          half_atol: absolute tolerance for float16.
          bfloat16_rtol: relative tolerance for bfloat16.
          bfloat16_atol: absolute tolerance for bfloat16.
          msg: Optional message to report on failure.
        """
        a = self._GetNdArray(a)
        b = self._GetNdArray(b)
        # types with lower tol are put later to overwrite previous ones.
        if (a.dtype == np.float32 or b.dtype == np.float32 or
                a.dtype == np.complex64 or b.dtype == np.complex64):
            rtol = max(rtol, float_rtol)
            atol = max(atol, float_atol)
        if a.dtype == np.float16 or b.dtype == np.float16:
            rtol = max(rtol, half_rtol)
            atol = max(atol, half_atol)
        if (a.dtype == tf.bfloat16.as_numpy_dtype or
                b.dtype == tf.bfloat16.as_numpy_dtype):
            rtol = max(rtol, bfloat16_rtol)
            atol = max(atol, bfloat16_atol)

        self.assertAllClose(a, b, rtol=rtol, atol=atol)

    def assertNotEqual(self, first, second, msg=None):
        if isinstance(first, (Tensor, Variable)):
            first = self.eval(first)
        if isinstance(second, (Tensor, Variable)):
            second = self.eval(second)
        super().assertNotEqual(first, second, msg)

    def assertAlmostEqual(self, first, second, places=None, msg=None,
                          delta=None):
        if isinstance(first, (Tensor, Variable)):
            first = self.eval(first)
        if isinstance(second, (Tensor, Variable)):
            second = self.eval(second)

        super().assertAlmostEqual(first, second, places, msg, delta)

    def assertAllClose(self, actual, desired, rtol=1e-7, atol=0., verbose=True):
        """ It compares the difference between actual and desired to
        ``atol + rtol * abs(desired)``.

        if actual or desired are Tensors, tries to evaluate them, so this
        should be called inside a ``with self.cashed_session():``

        Args:
            actual: array obtained
            desired: desired array value
            rtol: relative tolerance
            atol: absolute tolerance
            verbose: if True, the conflicting values are appended to the error message

        Raises:
            AssertionError: If actual and desired are not equal up to specified precision.

        """
        if isinstance(actual, (Tensor, Variable)):
            actual = self.eval(actual)
        if isinstance(desired, (Tensor, Variable)):
            desired = self.eval(desired)

        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, verbose=verbose)

    def assertAllEqual(self, actual, desired):
        return self.assertAllClose(actual, desired, rtol=0)

    def assertArrayEqual(self, actual, desired, verbose=True):
        """ Raises an AssertionError if two array_like objects or Tensors are not equal.

        Args:
            actual: array obtained
            desired: desired array
            verbose: if True, the conflicting values are appended to the error message

        Raises:
            AssertionError: If actual and desired are not equal.


        """
        if isinstance(actual, (Tensor, Variable)):
            actual = self.eval(actual)
        if isinstance(desired, (Tensor, Variable)):
            desired = self.eval(desired)

        np.testing.assert_array_equal(actual, desired, verbose=verbose)

    def assertArrayNotEqual(self, actual, desired):
        if isinstance(actual, (Tensor, Variable)):
            actual = self.eval(actual)
        if isinstance(desired, (Tensor, Variable)):
            desired = self.eval(desired)

        self.assertFalse(np.array_equal(actual, desired))

    @contextlib.contextmanager
    def _get_cached_session(self,
                            graph=None,
                            config=None,
                            use_gpu=False,
                            force_gpu=False,
                            crash_if_inconsistent_args=True):
        """See cached_session() for documentation."""
        if context.executing_eagerly():
            yield None
        else:
            if self._cached_session is None:
                sess = self._create_session(
                    graph=graph, config=config, use_gpu=use_gpu, force_gpu=force_gpu)
                self._cached_session = sess
                self._cached_graph = graph
                self._cached_config = config
                self._cached_use_gpu = use_gpu
                self._cached_force_gpu = force_gpu
                with self._constrain_devices_and_set_default(
                        sess, use_gpu, force_gpu) as constrained_sess:
                    yield constrained_sess
            else:
                if crash_if_inconsistent_args and self._cached_graph is not graph:
                    raise ValueError("The graph used to get the cached session is "
                                     "different than the one that was used to create the "
                                     "session. Maybe create a new session with "
                                     "self.session()")
                if crash_if_inconsistent_args and self._cached_config is not config:
                    raise ValueError("The config used to get the cached session is "
                                     "different than the one that was used to create the "
                                     "session. Maybe create a new session with "
                                     "self.session()")
                if crash_if_inconsistent_args and self._cached_use_gpu is not use_gpu:
                    raise ValueError(
                        "The use_gpu value used to get the cached session is "
                        "different than the one that was used to create the "
                        "session. Maybe create a new session with "
                        "self.session()")
                if crash_if_inconsistent_args and (self._cached_force_gpu is
                                                   not force_gpu):
                    raise ValueError(
                        "The force_gpu value used to get the cached session is "
                        "different than the one that was used to create the "
                        "session. Maybe create a new session with "
                        "self.session()")
                # If you modify this logic, make sure to modify it in _create_session
                # as well.
                sess = self._cached_session
                with self._constrain_devices_and_set_default(sess, use_gpu, force_gpu) as constrained_sess:
                    yield constrained_sess

    def _create_session(self, graph, config, use_gpu, force_gpu):
        """See session() for details."""
        if context.executing_eagerly():
            return None
        else:

            def prepare_config(config):
                """Returns a config for sessions.
                Args:
                  config: An optional config_pb2.ConfigProto to use to configure the
                    session.
                Returns:
                  A config_pb2.ConfigProto object.
                """
                if config is None:
                    config = config_pb2.ConfigProto()
                    config.allow_soft_placement = not force_gpu
                    config.gpu_options.per_process_gpu_memory_fraction = 0.3
                elif force_gpu and config.allow_soft_placement:
                    config = config_pb2.ConfigProto().CopyFrom(config)
                    config.allow_soft_placement = False
                # Don't perform optimizations for tests so we don't inadvertently run
                # gpu ops on cpu
                config.graph_options.optimizer_options.opt_level = -1
                config.graph_options.rewrite_options.constant_folding = (
                    rewriter_config_pb2.RewriterConfig.OFF)
                config.graph_options.rewrite_options.arithmetic_optimization = (
                    rewriter_config_pb2.RewriterConfig.OFF)
                return config

        return ErrorLoggingSession(graph=graph, config=prepare_config(config))

    def _clear_cached_session(self):
        if self._cached_session is not None:
            self._cached_session.close()
            self._cached_session = None

    @contextlib.contextmanager
    def cached_session(self,
                       graph=None,
                       config=None,
                       use_gpu=False,
                       force_gpu=False):
        """Returns a TensorFlow Session for use in executing tests.
        This method behaves differently than self.session(): for performance reasons
        `cached_session` will by default reuse the same session within the same
        test. The session returned by this function will only be closed at the end
        of the test (in the TearDown function).
        Use the `use_gpu` and `force_gpu` options to control where ops are run. If
        `force_gpu` is True, all ops are pinned to `/device:GPU:0`. Otherwise, if
        `use_gpu` is True, TensorFlow tries to run as many ops on the GPU as
        possible. If both `force_gpu and `use_gpu` are False, all ops are pinned to
        the CPU.
        Example:
        ```python
        class MyOperatorTest(test_util.TensorFlowTestCase):
          def testMyOperator(self):
            with self.cached_session(use_gpu=True) as sess:
              valid_input = [1.0, 2.0, 3.0, 4.0, 5.0]
              result = MyOperator(valid_input).eval()
              self.assertEqual(result, [1.0, 2.0, 3.0, 5.0, 8.0]
              invalid_input = [-1.0, 2.0, 7.0]
              with self.assertRaisesOpError("negative input not supported"):
                MyOperator(invalid_input).eval()
        ```
        Args:
          graph: Optional graph to use during the returned session.
          config: An optional config_pb2.ConfigProto to use to configure the
            session.
          use_gpu: If True, attempt to run as many ops as possible on GPU.
          force_gpu: If True, pin all ops to `/device:GPU:0`.
        Yields:
          A Session object that should be used as a context manager to surround
          the graph building and execution code in a test case.
        """
        if context.executing_eagerly():
            yield None
        else:
            with self._get_cached_session(
                    graph, config, use_gpu, force_gpu,
                    crash_if_inconsistent_args=True) as sess:
                yield sess

    @contextlib.contextmanager
    def _constrain_devices_and_set_default(self, sess, use_gpu, force_gpu):
        """Set the session and its graph to global default and constrain devices."""
        if context.executing_eagerly():
            yield None
        else:
            with sess.graph.as_default(), sess.as_default():
                if force_gpu:
                    # Use the name of an actual device if one is detected, or
                    # '/device:GPU:0' otherwise
                    gpu_name = gpu_device_name()
                    if not gpu_name:
                        gpu_name = "/device:GPU:0"
                    with sess.graph.device(gpu_name):
                        yield sess
                elif use_gpu:
                    yield sess
                else:
                    with sess.graph.device("/cpu:0"):
                        yield sess


def main():
    return unittest.main()

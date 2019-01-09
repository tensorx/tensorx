from unittest import TestCase
import numpy as np
from tensorx.data import itertools as itx
import itertools as it


class TestItertools(TestCase):

    def test_chunk_it(self):
        n_rows = 100

        data = np.arange(n_rows)
        it = itx.buffer_slice_it(data, n_rows, 3)

        for i in range(len(data)):
            data_j = next(it)
            self.assertEqual(data[i], data_j)

    def test_subset_chunk_it(self):
        n_rows = 100

        data = np.arange(n_rows)
        subset = range(50, 100)

        it = itx.subset_buffer_it(data, subset, 4)

        for i in subset:
            data_j = next(it)
            self.assertEqual(data[i], data_j)

    def test_divide_slice(self):
        subset = range(51, 100)

        sub_size = len(subset)
        divide_sub = itx.get_n_slices(sub_size, 3, subset.start)

        self.assertEqual(3, len(divide_sub))

    def test_ngrams(self):
        sentence = "hello there mr smith welcome back to the world"
        tokens = sentence.split()
        windows = itx.window_it(tokens, 3)
        # for window in windows:
        #    print(window)

        # print("fewer than ngram_size sequences")
        sentence = "hello there"

        tokens = sentence.split()
        windows = list(itx.window_it(tokens, 3))

        # print(windows)

        self.assertEqual(len(windows), 0)

        # for window in windows:
        #    print(window)

    def test_batch_it(self):
        num_samples = 6
        v = np.random.uniform(-1, 1, [num_samples, 2])
        padding = np.zeros([2])

        c_it = itx.buffer_slice_it(v, 6, buffer_size=3)
        # print(v)

        batch_size = 4
        b_it = itx.batch_it(c_it, batch_size, padding=True, fill_value=padding)

        for b in b_it:
            print(b)
            self.assertEqual(len(b), batch_size)
            # print(np.array(b))

        b_it = itx.batch_it(v, batch_size)
        last_batch = None
        try:
            for b in b_it:
                last_batch = b
                self.assertEqual(len(b), batch_size)

        except AssertionError:
            self.assertEqual(len(last_batch), num_samples % batch_size)

    def test_shuffle_it(self):
        v = list(range(10))
        padding = -1

        b_it = itx.batch_it(v, size=4, padding=True, fill_value=padding)

        s_it = itx.shuffle_it(b_it, 3)
        # for elem in s_it:
        #    print(elem)

    def test_reat_chunk_it(self):
        n_samples = 4
        repeat = 2
        v = np.random.uniform(0, 1, [n_samples, 1])
        data_it = itx.buffer_slice_it(v, buffer_size=2)

        def chunk_fn(x): return itx.buffer_slice_it(x, buffer_size=2)

        # for chunk in data_it:
        #    print(chunk)
        # print(data_it)
        data_it = itx.repeat_apply(chunk_fn, v, repeat)

        self.assertEqual(len(list(data_it)), n_samples * repeat)

    def test_chain_shuffle(self):
        n_samples = 4
        repeat = 2
        v = np.arange(0, n_samples, 1)
        data_it = itx.buffer_slice_it(v, buffer_size=2)

        def chunk_fn(x): return itx.buffer_slice_it(x, buffer_size=2)

        # first chain is normal, second is shuffled from the two repetitions
        data_it = itx.repeat_apply(chunk_fn, v, repeat)

        data_it = itx.chain_it(data_it, itx.shuffle_it(itx.repeat_apply(chunk_fn, v, repeat), buffer_size=8))

        data = list(data_it)

        unique_data = np.unique(data)
        counts = np.unique(np.bincount(data))

        self.assertEqual(len(unique_data), 4)
        self.assertEqual(len(counts), 1)
        self.assertEqual(counts[0], 4)

    def test_narrow_it(self):
        n_samples = 10
        t = 4
        data = range(n_samples)

        result1 = itx.narrow_it(data, t)
        self.assertEqual(len(list(result1)), t)

        result2 = itx.narrow_it(data, n_samples + t)
        self.assertEqual(len(list(result2)), n_samples)

    def test_take_it(self):
        n_samples = 8
        t = 5
        data = range(n_samples)
        result = itx.take_n_it(data, t)
        for r in result:
            print(r)

        # self.assertEqual(len(list(result)), t)

    def test_slice_it(self):
        n_samples = 10
        t = 3
        data = range(n_samples)

        it1 = itx.batch_it(data, t, padding=True, fill_value=0)
        it2 = itx.take_n_it(data, t)

        for s1 in it1:
            self.assertEqual(len(s1), t)

        for i, s2 in enumerate(it2):
            if i < 3:
                self.assertEqual(len(s2), 3)
            else:
                self.assertEqual(len(s2), 1)

    def test_bptt_it(self):
        n = 10000
        bsz = 5
        seq = 10
        data = np.arange(n, dtype=np.int32)

        data_it = itx.bptt_it(data,
                              batch_size=bsz,
                              seq_prob=1.0,
                              seq_len=seq,
                              enum=True)

        data_it1, data_it = it.tee(data_it, 2)

        for i, d in data_it1:
            self.assertEqual(i, 0)

        i, data1 = zip(*data_it)
        data1 = list(data1)
        last = data1[-1].flatten()[-1]
        num_batches = len(data1)
        self.assertEqual(last, n // bsz * bsz - 1)

        n = 0
        seq_sizes = 0
        for batch in data1:
            # print(batch)
            seq_sizes += np.shape(batch)[-1]
            n += 1

        avg_seq_len1 = seq_sizes / n

        data_it2 = itx.bptt_it(data,
                               batch_size=bsz,
                               seq_prob=1.0,
                               seq_len=seq,
                               num_batches=num_batches,
                               enum=True)

        # i, data2 = zip(*data_it2)

        n = 0
        seq_sizes = 0
        for i, batch in data_it2:
            # print(batch)
            seq_sizes += np.shape(batch)[-1]
            n += 1

        avg_seq_len2 = seq_sizes / n

        self.assertAlmostEqual(avg_seq_len1, avg_seq_len2, delta=1.0)

    def test_bptt_enum(self):
        """ test enumeration of each sequence when parallel sequences are buffered
        """
        data = range(20)
        data = map(str, data)

        data_it = itx.bptt_it(data,
                              batch_size=2,
                              seq_prob=1.0,
                              seq_len=3,
                              num_batches=2)
        print(next(data_it))
        print(next(data_it))
        print(next(data_it))
        print(next(data_it))

    def test_repeat_fn_exhaust(self):
        n_samples = 4
        v = np.random.uniform(0, 1, [n_samples, 1])
        data_it = itx.buffer_slice_it(v, buffer_size=2)

        self.assertEqual(len(list(data_it)), n_samples)

        # repeat 2 times
        def iter_fn(data): return itx.buffer_slice_it(data, buffer_size=2)

        data_it = itx.repeat_apply(iter_fn, v, 2)
        self.assertEqual(len(list(data_it)), n_samples * 2)

    def test_repeat_apply(self):
        # data = np.array(range(4))
        shape = [2, 2]
        data = np.random.uniform(0, 1, size=shape)

        data_3 = itx.repeat_apply(iter,[data], 3, enum=True)
        for i, d in data_3:
            np.testing.assert_array_equal(np.shape(d), shape)
            np.testing.assert_array_equal(d, data)

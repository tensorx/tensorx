""" Itertools
    TensorX utilities for iteration over data for common data iteration patterns such as::
        * batching
        * random shuffling
        * parallel sequence iteration
        * sliding windows

    These utilities work directly with python iterables and numpy arrays etc.
"""
import numpy as np
import itertools
from collections import deque


class Window:
    """ Window

    class defining a window _around_ a central element

    A window contains:
        a left []
        a target which is in the center of the window
        a right []
    """

    def __init__(self, left, target, right):
        self.left = left
        self.target = target
        self.right = right

    def __str__(self):
        return "(" + str(self.left) + "," + self.target + "," + str(self.right) + ")"


def windows(seq, window_size=1):
    """ Transforms a sequence into a sequence of overlapping windows
    Args:
        seq: an iterable
        window_size: size for the windows

    Returns:
        a list of Window
    """
    elem_indexes = range(0, len(seq))
    n = len(seq)

    ws = []
    # create a sliding window for each elem
    for w in elem_indexes:
        # lower limits
        wl = max(0, w - window_size)
        wcl = w

        # upper limits
        wch = n if w == n - 1 else min(w + 1, n - 1)
        wh = w + min(window_size + 1, n)

        # create window
        left = seq[wl:wcl]
        target = seq[w]
        right = seq[wch:wh]

        ws.append(Window(left, target, right))

    return ws


def pair_it(iterable):
    """ Pairs
    Iterates though a sequence showing each 2 items at a time

    Example::
        s -> (s0,s1), (s1,s2), (s2, s3), ...

    Returns:
        an iterable of tuples with each element paired with the next.
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def get_n_slices(n, n_slices=1, offset=0):
    """ get_n_slices

    returns m slices that divide an n-element sequence


    Args:
        n: number of elements
        n_slices: number of slices the n sequence is to be split into
        offset: starting point for the slices

    Returns:
        a list of slices that slice the n elements
    """
    len_split = int(n / n_slices)

    ss = [0]
    for s in range(len_split, len_split * n_slices, len_split):
        ss.append(s)

    ss.append(n)
    ranges = [slice(s[0] + offset, s[1] + offset, 1) for s in pair_it(ss)]

    return ranges


def narrow_it(iterable, n):
    """Iterates through iterable until n items are returns"""
    source_iter = iter(iterable)
    i = 0
    try:
        while i < n:
            yield next(source_iter)
            i += 1
    except StopIteration:
        return


def flatten_it(iterable):
    """Flatten one level of nesting in an iterator"""
    return itertools.chain.from_iterable(iterable)


def take_n_it(iterable, n):
    """Takes n items from iterable as a generator"""

    it = iter(iterable)
    while True:
        s = list(itertools.islice(it, n))
        if not s:
            return
        else:
            yield s


def advance_it(iterator, n):
    """ advance_it

    Advances the given iterator n steps, if n is None, consumes it entirely

    Args:
        iterator: and iterator to be advanced
        n: number of elements to advance
    """
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(itertools.islice(iterator, n, n), None)


def buffer_slice_it(seq, n_max=None, buffer_size=1, seq_range=None):
    """ buffer_slice_it

    iterates over items in data collections that can receive slices (seq[slice(start,stop,1)].
    The iterator returns one iteam at a time but it buffers one slice at a time.

    Use Cases:
        when slicing is cheap but reading one by one is not (e.g. hdf5 tables). Instead of
        using batch_it or repeated applications of take_it, we slice directly from
        a sequence type instance.

    Args:
        seq: a sequence type accepting slices (and len(seq) if n_max is not provided)
        n_max: maximum number of items to be retrieved
        buffer_size: maximum number of items to be loaded into memory at once
        seq_range: instead of n_max we can provide a range which will be used to slice the seq
                if n_max is passed and seq_range is not None, overwrite n_max with the size of the range
    Returns:
        a generator over the elements of input seq which are buffered
    """
    offset = 0
    if seq_range is not None:
        # overwrite n_max
        n_max = len(seq_range)
        offset = seq_range.start

        if buffer_size > n_max:
            buffer_size = n_max

    if n_max is None and seq_range is None:
        try:
            n_max = len(seq)
        except TypeError:
            raise TypeError("n_rows is None but dataset has no len()")

    n_chunks = n_max // buffer_size
    buffer_slices = get_n_slices(n_max, n_chunks, offset=offset)
    buffer_gen = (seq[s] for s in buffer_slices)
    row_gen = itertools.chain.from_iterable(buffer_gen)
    return row_gen


def window_it(iterable, n, as_list=True):
    """ window_it

    creates a fixed-sized sliding window iterator from the given iterable

    s -> (s0, ...,s(n-1)), (s1, ...,sn), (s2, ..., s(n+1)), ...
    """
    its = itertools.tee(iterable, n)
    for i, it in enumerate(its):
        advance_it(it, i)

    return map(list, zip(*its)) if as_list else zip(*its)


def batch_it(iterable, size, padding=False, fill_value=None):
    """batch_it

    Batches iterable and returns lists of elements with a given size

    Args:
        fill_value: the element to be used to pad the batches
        padding: if True, pads the last batch to be of at least the given size
        iterable: an iterable over data to be batched
        size: size of the batch to be returned

    Returns:
        a generator over batches with a given size, these can be smaller


    """
    batch_iter = take_n_it(iterable, size)
    try:
        while True:
            next_batch = next(batch_iter)
            if padding and len(next_batch) < size:
                padding_size = size - len(next_batch)
                next_batch.extend([fill_value] * padding_size)
            yield next_batch
    except StopIteration:
        return


def bptt_slice_it(n, seq_len, min_seq_len=2, seq_prob=0.95, overlap=0):
    """

    Args:
        overlap: if overlap, the last item will be overlapped with the first item of the next slice
        this is useful to create an iterator that returns the current sequence and the same sequence shifted by 1 into
        the future.
        n: number of items to slice (e.g. len(some_list) or np.shape(v)[-1])
        seq_len: base sequence length
        min_seq_len: minimum sequence length
        seq_prob: probability of base sequence length

    Note:
        Since the slices are probabilistic, there's no guarantee that the last slice would have at least min_seq_len, as
        such, if the last slice size is smaller than the minimum, the previous slice is extended to include the last
        elements as to avoid wasting data.

    Returns:
        an iterator over slices with variable size for back propagation through time-style learning, the idea is to slice
        a given dimension into segments of size seq_len with a given probability p and seq_len // 2 with probability
        1-p, to unbias the truncated back-propagation through time learning process.

    """
    if seq_prob > 1:
        raise ValueError("seq_prob has to be a value 0<x<=1.0")

    # average sequence lengths
    k1 = seq_len
    k2 = seq_len // 2

    i = 0
    while i < n:
        r = np.random.rand()
        if r < seq_prob:
            offset = np.random.normal(k1, min_seq_len)
        else:
            offset = np.random.normal(k2, min_seq_len)

        # prevents excessively small or negative sequence sizes
        # it also prevents excessively small splits at the end of a long sequence
        offset = max(min_seq_len, int(offset))

        if n - (i + offset) < min_seq_len:
            offset = n - i
        yield i, i + offset
        i += offset
        if overlap != 0 and i < n:
            i -= min(overlap, offset - 1)


def bptt_it(seq, batch_size, seq_len, min_seq_len=2, seq_prob=0.95, num_batches=None, enum=False, return_targets=False):
    """ Back Propagation Through Time Iterator

    Provides an iterator over a given sequence as batch_size independent parallel sequences,
    works in streaming data by buffering a certain number of batches at a time.

    Instead of using a fixed seq_len, it uses the base seq_len to return variable sized
    sequence batches. The algorithm using the parallel sequences should adjust the learning rate
    based on the sequence length to unbias que TBPTT:

    In summary it uses `N(seq_len,min_seq_len)` sequence length with probability `seq_prob`, and
    N(seq_len/2,min_seq_len) sequence length with probability 1 - seq-prob.

    References:
        - "Unbiasing Truncated Backpropagation Through Time", https://arxiv.org/abs/1705.08209
        - "Regularizing and Optimizing LSTM Language Models", https://arxiv.org/abs/1708.02182

    Args:
        return_targets: if true returns a tuple (seq,seq_targets). seq_targets is essentially seq shifted by 1 into
        the future. For example for a sequence [1,2,3,4,5,6,...] and sequence length of 4, returns ([1,2,3,4],[2,3,4,5])
        enum: if True returns tuples (seq_id, parallel_seq_batch)
        seq: iterable sequence
        batch_size: number of parallel sequences
        seq_len: base sequence length
        seq_prob: probability of base sequence
        min_seq_len: minimum sequence length
        num_batches: number of batches to be loaded at a time. If None consumes the entire data and loads it
            to load the entire iterable to memory
    Returns:
        a generator of (segment_id, parallel_seq_batch)
            - a segment id is an int that should be the same for sequential batches for the same buffer
            - the batches are numpy arrays with the shape [batch_size,seq_len]

        if num_batches is None returns only a generator of sequence batches, otherwise.
    """

    overlap = 0
    if return_targets:
        # we need to adjust to unbias since we'll be cutting the sequences to get the respective batches
        seq_len += 1
        min_seq_len += 1
        overlap = 1

    def to_batch(flat_data):
        n = np.shape(flat_data)[0]
        max_seq_len = n // batch_size
        # narrow remove items that don't fit into batch
        flat_data = flat_data[:max_seq_len * batch_size]
        batched_data = np.reshape(flat_data, [batch_size, max_seq_len])
        batched_data = np.ascontiguousarray(batched_data)
        return batched_data

    data = iter(seq)
    # load everything onto memory
    if num_batches is None:
        data = np.array(list(data))
        data = to_batch(data)
        data = iter([(0, data)])
    else:
        # average sequence lengths
        k1 = seq_len
        k2 = seq_len // 2

        # probability of sequence length distribution
        p1 = seq_prob
        p2 = 1 - p1

        # expected sequence length
        avg_seq_len = int(k1 * p1 + k2 * p2)
        # buffer max_seq * batches at a time
        buffer_size = batch_size * avg_seq_len * num_batches
        buffers = batch_it(data, buffer_size)
        data = ((i, to_batch(buffer)) for i, buffer in enumerate(buffers))

    # TODO not sure if data needs to be contiguous here, a view might be just fine
    # I can return a time-major batch of sequences which is contiguous

    batches = ((i, data_i[:, ii:fi]) for i, data_i in data
               for ii, fi in bptt_slice_it(n=np.shape(data_i)[-1],
                                           seq_len=seq_len,
                                           min_seq_len=min_seq_len,
                                           seq_prob=seq_prob,
                                           overlap=overlap))

    if return_targets:
        batches = ((i, d[:, :-1], d[:, 1:]) for i, d in batches)

    if not enum:
        if return_targets:
            batches = ((ctx, target) for _, ctx, target in batches)
        else:
            batches = (ctx for _, ctx in batches)

    return batches


def shuffle_it(seq, buffer_size):
    """ Shuffle iterator based on a buffer size

    Shuffling requires a finite list so we can use a buffer to build a list

    Args:
        seq: an iterable over a sequence
        buffer_size: the size of

    Returns:
        a shuffled sequence iterator

    """
    buffers = batch_it(seq, size=buffer_size)
    result = map(np.random.permutation, buffers)
    shuffled = itertools.chain.from_iterable((elem for elem in result))

    return shuffled


def chain_it(*data_it):
    """ Forward method for itertools chain

    To avoid unnecessary imports when using the "recipes" in views

    Args:
        *data_it: the iterables to be chained

    Returns:
        returns elements from the first iterable until exhausted, proceeds to the following iterables untill all are
        exhausted.

    """
    return itertools.chain(*data_it)


def chain_iterable(iterables, enum=False):
    # chain_iterable(['ABC', 'DEF'], enum=True) --> (0,A) (0,B) (0,C) (1,D) (1,E) (1,F)
    # chain_iterable(['ABC', 'DEF'], enum=False) --> A B C D E F
    for i, it in enumerate(iterables):
        for element in it:
            if enum:
                yield i, element
            else:
                yield element


def repeat_apply(fn, arg, n=1, enum=False):
    """ repeat_apply

    Repeats a function on the given data, n times

    Note:
        this intended to create iterators that cycle multiple times though
        data without having to copy the elements to cycle through them. If the
        fn returns a generator that iterates in a certain manner, this re-applies
        that same generator to the data. If however, data is an iterable that gets
        exhausted the first time it runs, this will return all the elements in the iterable
        just once.

    Args:
        arg: single argument to which the function is repeatedly applied
        fn : a function to be applied to the data
        n: number of times we iterate over iterable
        enum: if True, enumerates each repeat
    Returns:
        a generator on elements in data given by the fn it

    """

    it = (fn(arg) for fn in itertools.repeat(fn, n))
    it = chain_iterable(it, enum=enum)

    return it

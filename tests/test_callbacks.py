import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorx as tx
from tensorx.train.callbacks import *


def test_param_trigger():
    p1 = tx.Param(init_value=3, name="test")
    assert p1.n_units == 0
    assert p1.shape == []

    assert isinstance(p1, tx.Param)
    assert not isinstance(p1, Property)

    class Obs:
        def __init__(self):
            self.obj = None
            self.event_log = []

        def listen(self, obj):
            self.obj = obj
            obj.register(self)

        def trigger(self, event):
            self.event_log.append(event)
            assert isinstance(event, OnValueChange)

    obs = Obs()
    obs.listen(p1)
    p1.value = 3
    assert p1() == p1.value

    fn = p1.as_function()
    assert fn() == p1.value

    # triggers occur in set value only, and the graph works from a variable read
    p1.value = 4
    assert fn().numpy() == 4


def test_OnEveryStep():
    assert OnStep(2) != OnEveryStep(2)
    assert OnEveryStep(2).match(OnStep(2))
    assert OnStep(2).match(OnEveryStep(2))
    assert OnStep(2).match(OnEveryStep(1))
    assert OnEveryStep(1).match(OnStep(2))

    assert not OnStep(2).match(OnEveryStep(3))
    assert not OnEveryStep(3).match(OnEveryStep(1))

    assert OnEveryStep(1).match(OnEveryStep(3))
    assert OnStep(6, AT.END).match(OnEveryStep(3))
    assert OnEveryStep(3, AT.START).match(OnStep(6, AT.START))
    assert not OnEveryStep(3).match(OnStep(6, AT.START))
    assert OnEveryStep(3).match(OnStep(6, AT.END))


def test_OnEpoch():
    assert not OnEpoch(1).match(OnEpoch(2))
    assert OnEpoch(1).match(OnEveryEpoch(1))
    assert OnEveryEpoch(1).match(OnEpoch(1))
    assert OnEpoch(2).match(OnEveryEpoch(1))


def test_OnEveryEpochStep():
    assert not OnStep(2).match(OnEpochStep(2))
    assert not OnEpochStep(2).match(OnStep(2))
    assert not OnEpochStep(2).match(OnStep(2))
    assert OnStep(2) != OnEpochStep(4)
    assert OnEveryEpochStep(2).match(OnEpochStep(4))


def test_event_hash():
    event1 = OnValueChange("a")
    event2 = OnValueChange("b")

    d = {event1: 2}
    assert event1 in d
    assert event2 not in d

    event3 = OnEpoch(1)
    event4 = OnEveryEpoch(1)
    d = {event3: 2, event4: 3}
    assert event3 in d
    assert event4 in d

    event5 = OnStep(2)
    event6 = OnEveryStep(2)
    d = {event5: 1, event6: 2}
    assert event5 in d
    assert event6 in d

    event7 = OnLoop(at=AT.START)
    d = {event7: 1}
    assert event7 in d


def test_scheduler():
    prop_a = Property("a", 1)
    prop_b = Property("b", 1)
    prop_a2 = Property("a2", 1)
    prop_b2 = Property("b2", 1)

    changed = []

    scheduler = Scheduler(model=None, properties=[prop_a, prop_b, prop_a2, prop_b2])

    def fna(*args):
        assert len(args) == 2
        prop_a2.value = 2
        changed.append(1)

    def fnb(*args):
        assert len(args) == 2
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
    assert changed == [1, 2, 1]


def test_on_callback():
    a = Property("a", 1)
    c = Property("b", 2)

    def fn1(*args):
        assert len(args) == 2
        c.value -= 1

    def fn2(*args):
        assert len(args) == 2
        c.value *= 2

    def fn3(*args):
        assert len(args) == 2
        c.value *= 2

    cb1 = Callback({OnValueChange("a"): fn1}, priority=1)
    cb2 = Callback({OnCallback(cb1, at=AT.START): fn2}, priority=1)
    cb3 = Callback({OnCallback(cb1, at=AT.END): fn3}, priority=1)

    scheduler = Scheduler(model=None, properties=[a])
    scheduler.register(cb1)
    scheduler.register(cb2)
    scheduler.register(cb3)

    assert c.value == 2
    a.value = 2
    assert c.value == 6

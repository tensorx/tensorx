from enum import Enum
import itertools


class AT(Enum):
    """ AT Enum with the moments in a training loop step
    START
    ~
    STEP
    ~
    END
    """
    START = 0
    END = 2


class Event:
    __slots__ = []

    def __eq__(self, other):
        """

        :param other:
        :return:
        """
        if type(other) != self.__class__:
            return False
        else:
            attrs = [(getattr(self, attr), getattr(other, attr)) for attr in self.__slots__]
            return all(x is y for x, y in attrs)

    def match(self, other):
        return self.__eq__(other)

    def __hash__(self):
        cls = (self.__class__.__name__,)
        attrs = tuple(getattr(self, attr) for attr in self.__slots__)
        return hash(cls + attrs)

    def __str__(self):
        cls = self.__class__.__name__
        return "{cls}".format(cls=cls)


class OnTime(Event):
    """ OnTime Event represents an event that occurs at a given discrete step

    This represents a discrete step in a larger loop and further allows for the
    specification of the moment of that particular step

    Attributes:
        at (AT): step moment of the Event
        n (int): discrete step for the Event

    This is not meant to be triggered, instead the more specific ``OnStep``
    """
    __slots__ = ["at", "n"]

    def __init__(self, n: int = 1, at: Enum = AT.END):
        self.n = int(n)
        self.at = at

    def __str__(self):
        cls = self.__class__.__name__
        return "{cls}({n},{at})".format(cls=cls, n=self.n, at=self.at)


class OnStep(OnTime):
    def match(self, other):
        """ Matches OnStep with other.n % self.n == 0
        """
        if type(other) not in (OnEveryStep, OnStep):
            return False
        else:
            if type(other) == type(self):
                return self == other
            else:
                if self.at != other.at:
                    return False
                return self.n % other.n == 0


class OnEveryStep(OnStep):
    """ Event for moment on every n global step

    Example::
        n = 3
        for i in range(100)
            if i % n == 3:
                schedule.trigger(OnEverStep(3,AT.START))
            # step

            # end

    """

    def match(self, other):
        """ Matches OnStep with other.n % self.n == 0
        """
        if type(other) not in (OnEveryStep, OnStep):
            return False
        else:
            if self.at != other.at:
                return False
            return other.n % self.n == 0


class OnTrain(Event):
    """ Event with one of two moments in the training loop (``AT.START``, ``AT.END``)
    """
    __slots__ = ["at"]

    def __init__(self, at: Enum = AT.END):
        self.at = at

    def __str__(self):
        cls = self.__class__.__name__
        return "{cls}({at})".format(cls=cls, at=self.at)


class OnEpoch(OnTime):
    """ Event for specific epoch in the training

    Example::

        t = OnEpoch(2)
    """

    def match(self, other):
        """ Matches OnStep with other.n % self.n == 0
        """
        if type(other) not in (OnEveryEpoch, OnEpoch):
            return False
        else:
            if type(other) == type(self):
                return self == other
            else:
                if self.at != other.at:
                    return False
                return self.n % other.n == 0


class OnEveryEpoch(OnEpoch):
    """ Event for specific epoch in the training

    Example::

        t = OnEveryEpoch(2)
    """

    def match(self, other):
        """ Matches OnStep with other.n % self.n == 0
        """
        if type(other) not in (OnEveryEpoch, OnEpoch):
            return False
        else:
            if self.at != other.at:
                return False
            return other.n % self.n == 0


class OnEpochStep(OnTime):
    """ Event for one moment of a specific epoch step

    Example::
        t = OnEpochStep(3,AT.START)
    """

    def match(self, other):
        """ Matches OnStep with other.n % self.n == 0
        """
        if type(other) not in (OnEveryEpochStep, OnEpochStep):
            return False
        else:
            if type(other) == type(self):
                return self == other
            else:
                if self.at != other.at:
                    return False
                return self.n % other.n == 0


class OnEveryEpochStep(OnEpochStep):
    """ Event representing a moment on every n epoch step

    """

    def match(self, other):
        """ Matches OnStep with other.n % self.n == 0
        """
        if type(other) not in (OnEveryEpochStep, OnEpochStep):
            return False
        else:
            if self.at != other.at:
                return False
            return other.n % self.n == 0


class OnValueChange(Event):
    """ Event to be used with property value change events
    Attributes:
        name: name of the property to be monitored
    """
    __slots__ = ["name"]

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        cls = self.__class__.__name__
        return "{cls}({name})".format(cls=cls, name=self.name)


class Property:
    """ Property, an observable property with (name,value) that can register observers
    and notify them when a value changes

    observers must have the trigger method that accepts Trigger objects
    the only time this is not triggered is at object construction if a value is passed
    """

    def __init__(self, name, value=None):
        self.name = name
        self._value = value
        self.observers = []

    def register(self, obs):
        self.observers.append(obs)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        for observer in self.observers:
            observer.trigger(OnValueChange(self.name))


class StaticProperty(Property):
    def register(self, obs):
        pass

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class Callback:
    """ General Purpose Callback

    Callback objects can be registered at a scheduler to be executed on specific triggers
    they are also sortable by priority

    Args:
        trigger_dict: a dictionary mapping triggers to functions to be executed
        properties (List[str]): a list of properties required by this callback
        priority: int value which dictates callback execution priority (lower values take priority)
    """

    def __init__(self, trigger_dict, priority=1, properties=None):
        self.properties = properties
        self.trigger_dict: dict = trigger_dict
        self.priority = priority

    def __call__(self, trigger, model=None, properties=None):
        """ call receives an instance of the current model we're training
        along with the properties the callback is suposed to have access to

        it's the scheduler responsibility to pass these to the call

        Note:
            the model might be useful for computing and exposing new properties
            and the props are all the properties available to the callbacks

        Args:
            obj: an object giving access to model attributes and methods
            props: all the properties that can be monitored by callbacks

        Returns:
            a dictionary with property names and the respective values

        """
        # equals and hash might not match, because some general events match more specific ones
        # immutable objects are not the same but they match the same event: should I have a matches method ?
        fns = [self.trigger_dict[event] for event in self.trigger_dict.keys() if event.match(trigger)]
        return [fn(model, properties) for fn in fns]

    def __lt__(self, other):
        eq = self.priority - other.priority
        if not eq:
            return False
        return eq < 0


class Scheduler:
    def __init__(self, model, properties=[]):
        """

        Args:
            model: exposes a tensorx Model object
            properties: exposes a list of properties
            logs: exposes a dictionary with other logs that are not being tracked as properties
        """
        self.callbacks = []
        # stores priorities by event
        self.priority_cache = {}
        self.triggers = {}
        self.props = {prop.name: prop for prop in properties}

        for prop in self.props.values():
            prop.register(self)

        self.model = model

    def observe(self, prop):
        if prop.name not in self.props:
            prop.register(self)
            self.props[prop.name] = prop

    def register(self, callback):
        self.callbacks.append(callback)

        # update trigger map
        for trigger in callback.trigger_dict.keys():
            if trigger not in self.triggers:
                self.triggers[trigger] = [callback]
            else:
                self.triggers[trigger].append(callback)

        # update trigger cache
        for cached_trigger in self.priority_cache.keys():
            for trigger in callback.trigger_dict.keys():
                if cached_trigger.match(trigger):
                    cbs = self.priority_cache[cached_trigger]
                    self.priority_cache[cached_trigger] = list(sorted(cbs + [callback]))

        # creates and registers properties if callbacks specify outputs
        cb_props = callback.properties if callback.properties is not None else []
        for prop in cb_props:
            # register scheduler as listener
            prop.register(self)
            if prop.name in self.props:
                raise ValueError("{name} Property already in use".format(name=prop.name))
            self.props[prop.name] = prop

    def trigger(self, event: Event):
        """ Triggers an Event in the scheduler

        Args:
            event (Event): an event to be triggered by the scheduler to trigger the respective Callbacks
        """

        if not isinstance(event, Event):
            raise TypeError("Can only trigger the scheduler on Events, {} found".format(type(event)))

        matches = [self.triggers[trigger] for trigger in self.triggers.keys() if event.match(trigger)]

        if len(matches) > 0:
            matches = itertools.chain.from_iterable(matches)
            if event not in self.priority_cache:
                self.priority_cache[event] = list(sorted(matches))
            matches = self.priority_cache[event]

            for callback in matches:
                callback(event, self.model, self.props)

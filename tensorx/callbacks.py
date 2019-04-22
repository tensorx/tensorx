from enum import Enum


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

    def __init__(self, n: int, at: Enum = AT.END):
        self.n = n
        self.at = at

    def __str__(self):
        cls = self.__class__.__name__
        return "{cls}({n}{at})".format(cls=cls, n=self.n, at=self.at)


class OnStep(OnTime):
    pass


class OnTrain(Event):
    """ Event with one of two moments in the training loop (``AT.START``, ``AT.END``)
    """
    __slots__ = ["at"]

    def __init__(self, at: Enum = AT.END):
        self.at = at


class OnEpoch(OnTime):
    """ Event for specific epoch in the training

    Example::

        t = OnEpoch(2)
    """
    pass


class OnEveryEpoch(OnEpoch):
    """ Event for specific epoch in the training

    Example::

        t = OnEpoch(2)
    """

    def __eq__(self, other):
        """ Matches OnStep with other.n % self.n == 0
        """
        if type(other) not in (OnEveryEpoch, OnEpoch):
            return False
        else:
            if type(other) == type(self):
                # equals == all attributes must be the same
                attrs = [(getattr(self, attr), getattr(other, attr)) for attr in self.__slots__]
                return all(x is y for x, y in attrs)
            else:
                if self.at != other.at:
                    return False
                return other.n % self.n == 0


class OnEpochStep(OnTime):
    """ Event for one moment of a specific epoch step

    Example::
        t = OnEpochStep(3,AT.START)
    """
    pass


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

    def __eq__(self, other):
        """ Matches OnStep with other.n % self.n == 0
        """
        if type(other) not in (OnEveryStep, OnStep):
            return False
        else:
            if type(other) == type(self):
                # equals == all attributes must be the same
                attrs = [(getattr(self, attr), getattr(other, attr)) for attr in self.__slots__]
                return all(x is y for x, y in attrs)
            else:
                if self.at != other.at:
                    return False
                return other.n % self.n == 0


class OnEveryEpochStep(OnEpochStep):
    """ Event representing a moment on every n epoch step

    """

    def __eq__(self, other):
        """ Matches OnStep with other.n % self.n == 0
        """
        if type(other) not in (OnEveryEpochStep, OnEpochStep):
            return False
        else:
            if type(other) == type(self):
                # equals == all attributes must be the same
                attrs = [(getattr(self, attr), getattr(other, attr)) for attr in self.__slots__]
                return all(x is y for x, y in attrs)
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

    # def __eq__(self, other):
    #     return self.name == other.name
    #
    # def __hash__(self):
    #     return hash(self.name)


class Callback:
    """ General Purpose Callback

    Callback objects can be registered at a scheduler to be executed on specific triggers
    they are also sortable by priority

    Args:
        properties (List[Property]): a list of properties that will be created by this callback

    """

    def __init__(self, trigger: Event, properties=None, fn=None, priority=1):
        self.properties = properties
        self.trigger = trigger
        self.fn = fn
        self.priority = priority

    def __call__(self, model=None, props=None, logs=None):
        """ call receives an instance of the current model we're training
        along with the properties the callback is suposed to have access to

        it's the scheduler responsibility to pass these to the call

        Note:
            the model might be useful for computing and exposing new properties
            and the props are all the properties available to the callbacks

        Args:
            model: an object giving access to model attributes and methods
            props: all the properties that can be monitored by callbacks
            logs: values that are not monitored by callbacks anyway

        Returns:
            a list of properties and the affected values

        """
        if self.fn is not None:
            return self.fn(model)

    def __lt__(self, other):
        eq = self.priority - other.priority
        if not eq:
            return False
        return eq < 0


class Scheduler:
    def __init__(self, model, properties):
        """

        Args:
            model: exposes a tensorx Model object
            properties: exposes a list of properties
            logs: exposes a dictionary with other logs that are not being tracked as properties
        """
        self.callbacks = []
        # stores priorities by event
        self.priorities = {}
        self.props = {}

    def register(self, callback):
        self.callbacks.append(callback)
        self.priorities[callback.trigger] = []

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
        if len(self.priorities[event]) > 0:
            callbacks = self.priorities[event]
        else:
            callbacks = sorted(filter(lambda c: c.trigger == event, self.callbacks))
            self.priorities[event] = callbacks
        for cb in callbacks:
            cb()

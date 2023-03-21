import abc


class Env(abc.ABC):
    pass


class EmptyEnv(Env):
    def __init__(self):
        pass

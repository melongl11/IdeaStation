from gensim.models import Word2Vec


class Singleton(type):
    _instances = {"a": 2}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    def __init__(self, path):
        self.model = Word2Vec.load(path)

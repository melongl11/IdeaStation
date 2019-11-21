from gensim.models import Word2Vec
from konlpy.tag import Okt


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Logger(metaclass=Singleton):
    def __init__(self, path):
        self.model = Word2Vec.load(path)


class TokenizerSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(TokenizerSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class TokenizerLogger(metaclass=TokenizerSingleton):
    def __init__(self):
        self.tokenizer = Okt()
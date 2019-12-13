from gensim.models import Word2Vec
from konlpy.tag import Okt

import os

weights = [
    'wiki-window10.model',
    'news-window10.model',
    'kipris-window10.model',
    'news-wiki-window10.model',
    'wiki-kipris-window10.model',
    'news-kipris-window10.model',
    'news-wiki-kipris-window10.model'
]

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Logger(metaclass=Singleton):
    def __init__(self):
        self.models = [Word2Vec.load(os.path.dirname(os.path.abspath(__file__)) + '/../weight/' + weights[weight]) for weight in range(len(weights))]



class TokenizerSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(TokenizerSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class TokenizerLogger(metaclass=TokenizerSingleton):
    def __init__(self):
        self.tokenizer = Okt()
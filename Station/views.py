from django.shortcuts import render
from django.shortcuts import HttpResponse
from django.http import JsonResponse
from rest_framework import viewsets
from rest_framework import generics
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from .nlp.ensemble_k_means import *
from .nlp.elbow_k_means import *
from .nlp.correlation_clustering import *
from django.contrib.staticfiles.storage import staticfiles_storage

import json

import os
from django.core.files import File
# Create your views here.

weights = [
    'wiki-window10.model',
    'news-window10.model',
    'kipris-window10.model',
    'news-wiki-window10.model',
    'wiki-kipris-window10.model',
    'news-kipris-window10.model',
    'news-wiki-kipris-window10.model'
]

class Clusters(APIView):
    def get(self, request):
        key_word = request.query_params['word']
        weight = request.query_params['dataset']
        weight = int(weight)
        response_data = {}
        result = clustering(input_sentence=key_word, model_index=weight)
        response_data['n_cluster']=result[0]
        response_data['clusters']=result[1]
        response_data = json.dumps(response_data, ensure_ascii=False)
        return HttpResponse(response_data, content_type=u"application/json; charset=utf-8")


class ClustersWithN(APIView):
    def get(self, request):
        key_word = request.query_params['word']
        weight = request.query_params['dataset']
        weight = int(weight)
        print(key_word)
        key_word = key_word.split(' ')
        for i in range(len(key_word)):
            key_word[i] += '/N '
        key_word = ''.join(key_word)
        key_word = key_word[:len(key_word) - 1]
        print(key_word)
        response_data = {}
        result = clustering(input_sentence=key_word, model_index=weight)
        response_data['n_cluster'] = result[0]
        response_data['clusters'] = result[1]
        response_data = json.dumps(response_data, ensure_ascii=False)
        return HttpResponse(response_data, content_type=u"application/json; charset=utf-8")


class MandalArt(APIView):
    def get(self, request):
        words = request.query_params['word']
        weight = request.query_params['dataset']
        weight = int(weight)
        model_path = os.path.dirname(os.path.abspath(__file__)) + '/weight/' + weights[weight]
        result = make_mandal_art(input_sentence=words, model_index=weight)
        print(len(result))
        result = json.dumps(result, ensure_ascii=False)
        #result = ' '.join(result)
        return HttpResponse(result, content_type=u"application/json; charset=utf-8")


class MandalArtWithN(APIView):
    def get(self, request):
        key_word = request.query_params['word']
        weight = request.query_params['dataset']
        weight = int(weight)
        print(key_word)
        key_word = key_word.split(' ')
        for i in range(len(key_word)):
            key_word[i] += '/N '
        key_word = ''.join(key_word)
        key_word = key_word[:len(key_word) - 1]
        print(key_word)
        model_path = os.path.dirname(os.path.abspath(__file__)) + '/weight/model_namu_word2vec_size_100_window_10.model'
        result = make_mandal_art(input_sentence=key_word, model_index=weight)
        print(len(result))
        result = ' '.join(result)
        return HttpResponse(result, content_type=u"text/strings; charset=utf-8")
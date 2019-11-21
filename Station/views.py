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

class Clusters(APIView):
    def get(self, request):
        key_word = request.query_params['word']
        model_path = os.path.dirname(os.path.abspath(__file__)) + '/weight/namuwiki-2.model'
        response_data = {}
        result = clustering(input_sentence=key_word, model_path=model_path)
        response_data['n_cluster']=result[0]
        response_data['clusters']=result[1]
        response_data = json.dumps(response_data, ensure_ascii=False)
        return HttpResponse(response_data, content_type=u"application/json; charset=utf-8")
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .training import Predict

# Create your views here.

class MessageAPI(APIView):
    def get(self, request):
        emotion = Predict.predict(request.query_params.get('comment'), "raw")
        data = {"comment_emotion": emotion, "status": status.HTTP_200_OK}
        return Response(data, status=status.HTTP_200_OK)

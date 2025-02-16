from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .training import Predict

# Create your views here.

class HintAPI(APIView):
    def get(self, request):
        next_function = Predict.predict(
            request.query_params.get('user_id'),
            request.query_params.get('function_name'),
            request.query_params.get('time_slot'),
        )
        data = {"next_function": next_function, "status": status.HTTP_200_OK}
        return Response(data, status=status.HTTP_200_OK)

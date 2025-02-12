from django.urls import path, include
from apps.comment_emotion.views import MessageAPI

urlpatterns = [
    path("message", MessageAPI.as_view(), name="MessageAPI"),
]

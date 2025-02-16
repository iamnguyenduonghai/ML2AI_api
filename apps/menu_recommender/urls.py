from django.urls import path, include
from apps.menu_recommender.views import HintAPI

urlpatterns = [
    path("hint", HintAPI.as_view(), name="HintAPI"),
]

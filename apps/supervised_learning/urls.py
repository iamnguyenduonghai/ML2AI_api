from django.urls import path, include

urlpatterns = [
    path('linear_reg/', include('apps.supervised_learning.linear_reg.urls')),
]
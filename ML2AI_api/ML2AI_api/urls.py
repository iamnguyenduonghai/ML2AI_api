from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/supervised_learning/', include('apps.supervised_learning.urls')),
]

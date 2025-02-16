from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('app/api/comment-emotion/', include('apps.comment_emotion.urls')),
    path('app/api/menu-recommender/', include('apps.menu_recommender.urls')),
]

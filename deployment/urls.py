
from django.urls import path
from . import views

app_name = 'deployment'

urlpatterns = [
    path('', views.upload_video, name='upload_video'),
]

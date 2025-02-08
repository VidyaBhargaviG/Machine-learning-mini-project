from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('upload_dataset/', views.upload_dataset, name='upload_dataset'),
]

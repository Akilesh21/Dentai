# from django.urls import path
# from . import views

# urlpatterns = [
#     path('upload/', views.upload_image, name='upload_image'),
# ]

from django.urls import path
from .views import detect_cavity

urlpatterns = [
    path('', detect_cavity, name='detect_cavity'),  # Update to point to the detect_cavity view
]

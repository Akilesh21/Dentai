from django.urls import path
from .views import detect_disease,upload_success,home

urlpatterns = [
    path('',home,name='home'),
    path('upload/success/', upload_success, name='upload_success'),
    path('detect', detect_disease, name='detect_disease'),
    
    
]

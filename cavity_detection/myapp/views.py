# views.py
from django.shortcuts import render, redirect
from .forms import ImageUploadForm
from django.http import HttpResponse
from django.http import HttpResponse
from PIL import Image
import numpy as np 
import tensorflow as tf
import torch 
import torchvision.transforms as transforms  
model = tf.keras.models.load_model('C:/Users/akhil/Desktop/Major_Project/teeth_model.h5')  

def detect_disease(request):
    img_width = 150
    img_height = 150
    if request.method == 'POST':
        print(request.POST)  # Inspect the contents of request.POST
        print(request.FILES) 
    if request.method == 'POST' and request.FILES['image']:
        image_file = request.FILES['image']
        image = Image.open(image_file)

        # Preprocess the image (replace with your preprocessing steps)
        image = image.resize((img_width, img_height))  # Assuming you have img_width, img_height defined
        image_array = np.asarray(image)
        image_array = image_array / 255.0  # Example normalization
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Pass the image to the model for prediction
        prediction = model.predict(image_array)

        # Process the prediction result (e.g., get class, confidence)
        class_index = np.argmax(prediction) 
        confidence = prediction[0][class_index]
        if class_index == 0:
            disease_class = 'caries'
        else:
            disease_class = 'non-caries'

        return render(request, 'main.html', {
                'prediction': disease_class,
                'confidence': confidence
              })
    else:
        form = ImageUploadForm()
        return render(request, 'main.html', {'form': form})

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('upload_success')
    else:
        form = ImageUploadForm()
    return render(request, 'main.html', {'form': form})

def upload_success(request):
    return render(request, 'upload.html')

def home(request):
    return render(request,'home.html')

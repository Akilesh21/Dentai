from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .utils import predict_image, show_image_with_boxes
from PIL import Image
import torch
import numpy as np
import io
import base64

def load_complete_model(model_path):
    # Load the entire model
    model = torch.load(model_path,map_location=torch.device('cpu'))
    # Set the model to evaluation mode
    model.eval()
    return model


# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_complete_model("C:/Users/akhil/Desktop/Major_Project/x_ray_model.pth")  # Load your trained model here

def detect_cavity(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        # Save the uploaded image
        fs = FileSystemStorage()
        image_path = fs.save(uploaded_image.name, uploaded_image)
        # Perform cavity detection on the uploaded image
        pred_boxes, pred_labels, pred_scores = predict_image(model, device,  fs.path(image_path))
        original_image = Image.open(image_path).convert("RGB")
        # Call the visualization function
        processed_image = show_image_with_boxes(np.array(original_image), pred_boxes, pred_labels, pred_scores)
        return render(request, 'result.html',  {'processed_image': processed_image})
    return render(request, 'upload_form.html')
        # # Call the visualization function and get the processed image
        # processed_image = show_image_with_boxes(np.array(original_image), pred_boxes, pred_labels, pred_scores)
        # # Convert the processed image to base64
        # buffered = io.BytesIO()
        # processed_image.savefig(buffered, format='png')
        # buffered.seek(0)
        # img_str = base64.b64encode(buffered.read()).decode()
        # return render(request, 'result.html', {'processed_image': img_str})

from django.shortcuts import render
from django.http import HttpResponse

def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        # Process the uploaded image here
        return HttpResponse("Image uploaded successfully")
    else:
        return render(request, 'upload_form.html')  # Render a form for uploading the image
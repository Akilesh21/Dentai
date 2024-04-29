# from PIL import Image
# import torch
# import numpy as np
# import torchvision.transforms.functional as F
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import io
# import base64

# def predict_image(model, device, image_path, target_size=(224, 224), confidence_threshold=0.8):
#     # Load image and preprocess
#     image = Image.open(image_path).convert("RGB")
#     image = F.resize(image, target_size)
#     image = image.resize(target_size, Image.Resampling.LANCZOS)
#     image = F.to_tensor(image).unsqueeze(0).to(device)  # Add batch dimension and send to device

#     model.eval()  # Set model to evaluation mode
#     with torch.no_grad():
#         predictions = model(image)
    
#     # Process predictions
#     pred_boxes = predictions[0]['boxes'].cpu().numpy()
#     pred_labels = predictions[0]['labels'].cpu().numpy()
#     pred_scores = predictions[0]['scores'].cpu().numpy()
    
#     # Filter predictions based on confidence threshold
#     pred_boxes = pred_boxes[pred_scores >= confidence_threshold].astype(np.int32)
#     pred_labels = pred_labels[pred_scores >= confidence_threshold]
#     pred_scores = pred_scores[pred_scores >= confidence_threshold]

#     return pred_boxes, pred_labels, pred_scores

# def show_image_with_boxes(image, boxes, labels, scores, label_names={1: "Caries", 2: "Other"}):
#     # Create figure and axes
#     fig, ax = plt.subplots(1)

#     # Display the image
#     ax.imshow(image)
    
#     # Create a Rectangle patch for each box and add it to the plot
#     for box, label, score in zip(boxes, labels, scores):
#         x, y, width, height = box
#         rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)
#         plt.text(x, y, f'{label_names.get(label, label)}: {score:.2f}', color='white', fontsize=8,
#                  bbox=dict(facecolor='red', alpha=0.5, pad=0, edgecolor='none'))
#     # Save the plot to a byte array
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     image_data = base64.b64encode(buf.getvalue()).decode()
#     plt.close()

#     return image_data
#     plt.show()

import io
import base64
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F

def predict_image(model, device, image_path, target_size=(224, 224), confidence_threshold=0.8):
    image = Image.open(image_path).convert("RGB")
    image = F.resize(image, target_size)
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    image = F.to_tensor(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(image)
    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()
    pred_boxes = pred_boxes[pred_scores >= confidence_threshold].astype(np.int32)
    pred_labels = pred_labels[pred_scores >= confidence_threshold]
    pred_scores = pred_scores[pred_scores >= confidence_threshold]
    return pred_boxes, pred_labels, pred_scores

def show_image_with_boxes(image, boxes, labels, scores, label_names={1: "Caries", 2: "Other"}):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for box, label, score in zip(boxes, labels, scores):
        x, y, width, height = box
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y, f'{label_names.get(label, label)}: {score:.2f}', color='white', fontsize=8,
                 bbox=dict(facecolor='red', alpha=0.5, pad=0, edgecolor='none'))
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

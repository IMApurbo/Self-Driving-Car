import torch
import cv2
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt

# Load the YOLOP model from the official repository
model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define the preprocessing function
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((640, 640)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_frame(frame):
    # Preprocess frame
    input_image = transform(frame).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        det_out, da_seg_out, ll_seg_out = model(input_image)

    # Convert outputs to numpy arrays
    da_seg_out_np = da_seg_out[0][0].detach().cpu().numpy()
    ll_seg_out_np = ll_seg_out[0][0].detach().cpu().numpy()

    # Display the original frame
    cv2.imshow('Original Frame', frame)

    # Drivable Area Segmentation visualization
    cv2.imshow('Drivable Area Segmentation', da_seg_out_np)

    # Lane Line Segmentation visualization
    cv2.imshow('Lane Line Segmentation', ll_seg_out_np)

# Start webcam capture
cap = cv2.VideoCapture(0)  # Use the default camera (0)

if not cap.isOpened():
    print("Cannot access the camera.")
    exit()

print("Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB format for PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame and display results
    process_frame(frame_rgb)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

import torch
import cv2
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F

# Load the YOLOP model from the official repository
model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define the preprocessing function
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((320, 320)),  # Reduce the input size for better performance
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

    # Resize outputs to match the original frame dimensions
    da_seg_out_resized = cv2.resize(da_seg_out_np, (frame.shape[1], frame.shape[0]))
    ll_seg_out_resized = cv2.resize(ll_seg_out_np, (frame.shape[1], frame.shape[0]))

    # Normalize outputs to the range [0, 255]
    da_seg_out_resized = (da_seg_out_resized * 255).astype(np.uint8)
    ll_seg_out_resized = (ll_seg_out_resized * 255).astype(np.uint8)

    # Create a combined output image
    combined_output = np.zeros_like(frame)
    combined_output[:, :, 1] = da_seg_out_resized  # Green channel for Drivable Area Segmentation
    combined_output[:, :, 2] = ll_seg_out_resized  # Red channel for Lane Line Segmentation

    # Display images
    cv2.imshow('Original Frame', frame)
    cv2.imshow('YOLOP Output', combined_output)

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

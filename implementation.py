import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('hand_gesture_m.h5')

# Open the camera
video_capture = cv2.VideoCapture(0)  # 0 for default camera, adjust if multiple cameras

# Check if the camera was opened successfully
if not video_capture.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set the frame width and height (adjust as needed)
frame_width = 640
frame_height = 480

# Set the input size expected by your model
input_size = (128, 128)  # Adjust based on your model's input requirements

# Function to preprocess frame
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, input_size)
    normalized_frame = resized_frame / 255.0  # Normalize the frame
    return np.expand_dims(normalized_frame, axis=0)

# Function to get class labels
class_labels = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '+', '-', '*', '/', '/','confirm', '**', '%', 'clear']

# Read frames from the camera
while video_capture.isOpened():
    ret, frame = video_capture.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Mirror the frame (optional)
    frame = cv2.flip(frame, 1)
    
    # Preprocess the frame
    input_data = preprocess_frame(frame)
    
    # Make prediction
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    
    # Optionally, post-process prediction if needed
    
    # Display the frame with prediction
    cv2.putText(frame, f'Predicted: {class_labels[predicted_class]} ({confidence:.2f})',
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Camera', frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
video_capture.release()
cv2.destroyAllWindows()

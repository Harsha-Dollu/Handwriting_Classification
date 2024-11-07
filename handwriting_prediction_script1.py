import cv2
import numpy as np
import tensorflow as tf

# Function to preprocess a single image
def preprocess_image(image_path, size=(128, 128)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    image = cv2.resize(image, size)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (for grayscale)
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    return image

# Function to predict handwriting quality
def predict_handwriting(image_path, model_path='improved_handwriting_modeL.h5'):
    model = tf.keras.models.load_model(model_path)
    
    # Check model input shape
    print(f"Model expects input shape: {model.input_shape}")
    
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    score = prediction[0][0]  # Get the score from the prediction array
    
    # Define a threshold for classification
    threshold = 0.5  # Adjust the threshold as needed
    label = 'Good Handwriting' if score > threshold else 'Bad Handwriting'
    
    return label, score

# Test the prediction function
image_path1 = r'C:\Users\harsh\OneDrive\Desktop\python\python_project1\textImage6.jpeg'  
image_path2 = r'C:\Users\harsh\OneDrive\Desktop\python\python_project1\download.jpg'  

result1 = predict_handwriting(image_path1)
result2 = predict_handwriting(image_path2)

print(f"Prediction: {result1[0]}, Actual Score: {result1[1]:.4f}")
print(f"Prediction: {result2[0]}, Actual Score: {result2[1]:.4f}")

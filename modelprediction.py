import os
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the saved model
model = load_model('seedling_classifier.h5')  # Adjust filename if necessary

# Load class indices from JSON file
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Reverse the mapping for easy lookup
class_indices_reversed = {v: k for k, v in class_indices.items()}

# Function to preprocess the image for prediction
def prepare_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(150, 150))  # Resize to model's input size
        img_array = image.img_to_array(img) / 255.0  # Convert to array and scale
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

while True:
    # Get the image file path from user input
    img_path = input("Enter the path of the PNG image for prediction (or type 'o' to exit): ")

    # Exit condition
    if img_path.lower() in ['o', 'exit']:
        print("Exiting the program.")
        break

    # Print the path for debugging
    print(f"User entered path: {img_path}")

    # Check if the file exists and is a PNG image
    if not os.path.isfile(img_path) or not img_path.lower().endswith('.png'):
        print("The specified file does not exist or is not a PNG image. Please try again.")
        continue

    # Prepare the image for prediction
    prepared_image = prepare_image(img_path)
    
    if prepared_image is None:
        continue  # Skip to the next iteration if there was an error

    # Make prediction
    predictions = model.predict(prepared_image)
    predicted_class = np.argmax(predictions)

    # Get the predicted class name safely
    predicted_class_name = class_indices_reversed.get(predicted_class, "Unknown class")

    # Output the results
    print(f"Image: {os.path.basename(img_path)}")
    print(f"Predicted class index: {predicted_class}")
    print(f"Predicted class name: {predicted_class_name}\n")

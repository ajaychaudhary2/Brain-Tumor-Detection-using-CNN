import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load SavedModel
loaded_model = tf.saved_model.load('model/saved_model')
infer = loaded_model.signatures['serving_default']

# Define class names
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

def predict_tumor(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Convert to tensor
    input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Predict using the model
    prediction = infer(input_tensor)
    output = list(prediction.values())[0].numpy()

    # Class and confidence
    predicted_index = np.argmax(output)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(output)) * 100

    # Format result message
    if predicted_class == 'no_tumor':
        message = "No tumor in brain"
    else:
        message = f"⚠️ Tumor Detected: {predicted_class.capitalize()} tumor"

    return predicted_class, round(confidence, 2), message

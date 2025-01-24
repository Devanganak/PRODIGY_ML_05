import pickle
import numpy as np

# Load the saved model
with open('model.pkl', 'rb') as f:
    classifier, regressor, label_encoder = pickle.load(f)

# Example: Using the classifier to make a prediction
sample_image = np.random.rand(128, 128, 3).flatten()  # Example random image, replace with actual image data
sample_label = classifier.predict([sample_image])  # Make prediction

# Decode the label (fruit name) from the encoded label
decoded_label = label_encoder.inverse_transform(sample_label)

# Example: Using the regressor to predict calorie estimation
sample_calorie = regressor.predict([sample_image])  # Make prediction for calories

print(f"Predicted fruit: {decoded_label[0]}")
print(f"Predicted calorie: {sample_calorie[0]} kcal")

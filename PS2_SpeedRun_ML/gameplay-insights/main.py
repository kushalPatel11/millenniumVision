import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_preprocessing import load_and_preprocess_data
from insight_model import build_model
import tensorflow as tf

# Load data
X_scaled, y, scaler = load_and_preprocess_data('updated_gameplay_stats.csv')

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Build the model
model = build_model(input_shape=(X_train.shape[1],), output_classes=len(np.unique(y_encoded)))

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc}")

# Example of predicting insights based on new gameplay data
new_gameplay_stats = pd.DataFrame({
    'headshots': [5],
    'shots_hit': [20],
    'shots_missed': [15],
    'accuracy': [90],
    'crosshair_placement': [3],
    'spray_control': [4]
})

# Feature engineering for new data
new_gameplay_stats['headshot_ratio'] = new_gameplay_stats['headshots'] / (new_gameplay_stats['shots_hit'] + new_gameplay_stats['shots_missed'])
new_gameplay_stats['missed_shot_ratio'] = new_gameplay_stats['shots_missed'] / (new_gameplay_stats['shots_hit'] + new_gameplay_stats['shots_missed'])
new_gameplay_stats['accuracy_trend'] = new_gameplay_stats['accuracy']

# Standardize the new data
X_new_scaled = scaler.transform(new_gameplay_stats[['headshot_ratio', 'missed_shot_ratio', 'accuracy_trend', 'crosshair_placement', 'spray_control']])

# Make a prediction
prediction = model.predict(X_new_scaled)
predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
print("Generated Insight for Improvement: ", predicted_class[0])

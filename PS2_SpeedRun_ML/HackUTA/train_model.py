import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore

# Define input dimensions
input_frames = 10  # Number of frames to feed into the model
input_height = 240  # Height of each frame
input_width = 320  # Width of each frame

# Data generator class for loading video data
class VideoDataset(tf.keras.utils.Sequence):
    def __init__(self, video_folder, batch_size=4):
        self.video_folder = video_folder
        self.video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4') or f.endswith('.webm')]
        self.labels = [0 if 'missed' in f else 1 for f in self.video_files]  # 0: Missed, 1: Hit
        self.batch_size = batch_size

    def __len__(self):
        return len(self.video_files) // self.batch_size

    def __getitem__(self, idx):
        batch_files = self.video_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_data = []
        for video_file in batch_files:
            video_path = os.path.join(self.video_folder, video_file)
            frames = self.load_video_frames(video_path)
            batch_data.append(frames)

        return np.array(batch_data), np.array(batch_labels)

    def load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while frame_count < input_frames:
            ret, frame = cap.read()
            if not ret:
                break  # End of video or error in reading frame

            frame = cv2.resize(frame, (input_width, input_height))  # Resize frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frames.append(frame)
            frame_count += 1

        cap.release()

        # If fewer than input_frames, repeat the last frame
        while len(frames) < input_frames:
            frames.append(frames[-1])

        frames = np.stack(frames, axis=0)  # Shape: [input_frames, height, width, 3]
        return frames

# Define the model using TensorFlow
def build_model():
    model = models.Sequential()

    # 3D convolutional layer (for time-series + spatial data)
    model.add(layers.Conv3D(16, (3, 3, 3), activation='relu', input_shape=(input_frames, input_height, input_width, 3)))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(layers.Conv3D(32, (3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))  # 2 output classes: Missed (0) and Hit (1)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Training the model
def train_model(video_folder, epochs=10, batch_size=4):
    dataset = VideoDataset(video_folder, batch_size)
    model = build_model()

    # Train the model using the dataset
    model.fit(dataset, epochs=epochs, batch_size=batch_size)

    print("Training finished.")
    return model

# Example usage
video_folder = './Dataset/'  # Replace with the actual path
trained_model = train_model(video_folder, epochs=10, batch_size=4)

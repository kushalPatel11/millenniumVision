import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
import math
from PIL import Image

# Define the dataset class
class VideoDataset(Dataset):
    def __init__(self, base_folder, input_frames, transform=None):
        self.headshots_folder = os.path.join(base_folder, 'headshots')
        self.shots_hit_folder = os.path.join(base_folder, 'shots_hit')
        self.shots_missed_folder = os.path.join(base_folder, 'shots_missed')
        self.spray_control_folder = os.path.join(base_folder, 'spray_control')

        self.transform = transform
        self.input_frames = input_frames

        # Load video files from each folder
        self.headshot_files = [os.path.join(self.headshots_folder, f) for f in os.listdir(self.headshots_folder) if f.endswith('.mp4') or f.endswith('.webm')]
        self.shots_hit_files = [os.path.join(self.shots_hit_folder, f) for f in os.listdir(self.shots_hit_folder) if f.endswith('.mp4') or f.endswith('.webm')]
        self.shots_missed_files = [os.path.join(self.shots_missed_folder, f) for f in os.listdir(self.shots_missed_folder) if f.endswith('.mp4') or f.endswith('.webm')]
        self.spray_control_files = [os.path.join(self.spray_control_folder, f) for f in os.listdir(self.spray_control_folder) if f.endswith('.mp4') or f.endswith('.webm')]

        self.video_files = self.headshot_files + self.shots_hit_files + self.shots_missed_files + self.spray_control_files
        self.labels = self.generate_labels()

    def generate_labels(self):
        labels = []
        for _ in self.headshot_files:
            labels.append(self.create_label(headshots=1, shots_hit=1, shots_missed=0, spray_control=1))
        for _ in self.shots_hit_files:
            labels.append(self.create_label(headshots=0, shots_hit=1, shots_missed=0, spray_control=1))
        for _ in self.shots_missed_files:
            labels.append(self.create_label(headshots=0, shots_hit=0, shots_missed=1, spray_control=1))
        for _ in self.spray_control_files:
            labels.append(self.create_label(headshots=0, shots_hit=1, shots_missed=0, spray_control=1))
        return labels

    def create_label(self, headshots, shots_hit, shots_missed, spray_control):
        shots_fired = shots_hit + shots_missed
        crosshair_placement = math.floor(shots_hit / shots_fired) if shots_fired > 0 else 0
        accuracy = (headshots + shots_hit + shots_missed + shots_fired + crosshair_placement + spray_control) / 6
        accuracy = round(accuracy, 2)
        return torch.tensor([headshots, shots_hit, shots_missed, shots_fired, crosshair_placement, spray_control, accuracy], dtype=torch.float32)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)

        frames = []
        to_tensor = transforms.ToTensor()  # Transformation to convert PIL image to tensor

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (input_width, input_height))  # Resize frame

            # Convert numpy array to PIL image
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to RGB and PIL format

            if self.transform:
                frame = self.transform(frame)  # Apply transformations if any

            frame = to_tensor(frame)  # Convert PIL image to tensor
            frames.append(frame)

        cap.release()

        # Stack frames along the first dimension (channels)
        frames = torch.stack(frames)  # Combine list of frames into a tensor

        # Ensure fixed number of input frames
        if frames.size(0) < self.input_frames:
            # If fewer frames, pad with zeros
            padding = (0, 0, 0, 0, 0, self.input_frames - frames.size(0))
            frames = F.pad(frames, padding, "constant", 0)
        else:
            frames = frames[:self.input_frames]  # Select first `input_frames`

        return frames, self.labels[idx]


# Define the model with dropout to prevent overfitting
class VideoRatingModel(nn.Module):
    def __init__(self, num_ratings, input_frames, input_height, input_width):
        super(VideoRatingModel, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(0.2),  # Dropout to prevent overfitting
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(0.2)  # Dropout to prevent overfitting
        )

        self.flattened_size = self._get_flattened_size(input_frames, input_height, input_width)
        self.fc = nn.Linear(self.flattened_size, num_ratings)

    def _get_flattened_size(self, input_frames, input_height, input_width):
        x = torch.zeros(1, 3, input_frames, input_height, input_width)
        x = self.conv3d(x)
        return int(torch.prod(torch.tensor(x.shape[1:])))

    def forward(self, x):
        x = self.conv3d(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x) * 10  # Outputs scaled to [0, 10]


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_frames = 3  # Number of frames to input to the model (set to 3)
input_height = 240  # Height of each frame
input_width = 320  # Width of each frame
batch_size = 8
num_ratings = 7  # 7 attributes: headshots, shots_hit, shots_missed, shots_fired, crosshair_placement, spray_control, accuracy

# Initialize model
model = VideoRatingModel(num_ratings, input_frames, input_height, input_width).to(device)

# Load dataset with transformations for augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Random flip to augment the dataset
    transforms.RandomRotation(10),  # Random rotation to augment the dataset
])

video_dataset = VideoDataset(base_folder='Dataset', input_frames=input_frames, transform=transform)
dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True)

# Loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop with Early Stopping
best_loss = float('inf')
patience = 5  # Stop if no improvement for 5 epochs
early_stop_counter = 0

results = []
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for videos, labels in dataloader:
        videos, labels = videos.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)  # Forward pass
        loss = criterion(outputs, labels)  # MSELoss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Early Stopping Logic
    if avg_loss < best_loss:
        best_loss = avg_loss
        early_stop_counter = 0
        # Save the best model
        # torch.save(model.state_dict(), 'best_video_rating_model.pth')
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

    # Store actual results for CSV
    with torch.no_grad():
        for videos, labels in dataloader:
            videos = videos.to(device)
            outputs = model(videos).detach().cpu().numpy()
            results.extend(outputs)

# Save results to CSV
results_df = pd.DataFrame(results, columns=['headshots', 'shots_hit', 'shots_missed', 'shots_fired', 'crosshair_placement', 'spray_control', 'accuracy'])
results_df.to_csv('video_rating_actual_results.csv', index=False)
print("Results saved to video_rating_actual_results.csv")

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import math

# Define the dataset class
class VideoDataset(Dataset):
    def __init__(self, video_folder, input_frames, transform=None):
        self.video_folder = video_folder
        self.transform = transform
        self.video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4') or f.endswith('.webm')]
        
        # Assuming ground truth ratings for attributes are stored in the filenames
        # Example filename format: video_hit_7_8_5.mp4 (7: shots hit, 8: shots missed)
        self.labels = [self.extract_ratings(f) for f in self.video_files]
        self.input_frames = input_frames

    def extract_ratings(self, filename):
        # Assuming ratings are embedded in the filename as `video_hit_7_8_5.mp4`
        parts = filename.split('_')
        shots_hit = int(parts[-3])
        shots_missed = int(parts[-2])
        
        # Calculate the derived attributes
        shots_fired = shots_hit + shots_missed
        crosshair_placement = math.floor(shots_hit / shots_fired) if shots_fired > 0 else 0
        
        # Accuracy is the average of all attributes on a scale from 1 to 10
        accuracy = (shots_hit + shots_missed + shots_fired + crosshair_placement) / 4
        accuracy = round(accuracy, 2)

        # Convert to torch tensor
        return torch.tensor([shots_hit, shots_missed, shots_fired, crosshair_placement, accuracy], dtype=torch.float32)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.video_folder, self.video_files[idx])
        cap = cv2.VideoCapture(video_path)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (input_width, input_height))  # Resize frame
            if self.transform:
                frame = self.transform(frame)  # Apply transformations if any
            frames.append(frame)

        cap.release()

        # Normalize the pixel values
        frames = np.array(frames) / 255.0  # Normalize to range [0, 1]

        # Ensure the shape is (num_frames, channels, height, width)
        frames = frames.transpose((0, 3, 1, 2))  # Change to (num_frames, 3, height, width)
        frames = torch.tensor(frames).float()  # Convert to float tensor

        # Ensure fixed number of input frames
        if frames.size(0) < self.input_frames:
            frames = F.pad(frames, (0, 0, 0, 0, 0, self.input_frames - frames.size(0)), "constant", 0)
        else:
            frames = frames[:self.input_frames]  # Select first `input_frames`

        return frames, self.labels[idx]


# Define the multi-task 3D CNN model
class VideoRatingModel(nn.Module):
    def __init__(self, num_ratings, input_frames, input_height, input_width):
        super(VideoRatingModel, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )

        self.flattened_size = self._get_flattened_size(input_frames, input_height, input_width)
        self.fc = nn.Linear(self.flattened_size, num_ratings)  # Output one rating per attribute

    def _get_flattened_size(self, input_frames, input_height, input_width):
        x = torch.zeros(1, 3, input_frames, input_height, input_width)  # Dummy input tensor
        x = self.conv3d(x)
        return int(np.prod(x.shape[1:]))  # Flatten size

    def forward(self, x):
        x = self.conv3d(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return torch.sigmoid(x) * 10  # Output scaled to [0, 10]


# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_frames = 16  # Number of frames to input to the model
input_height = 240  # Height of each frame
input_width = 320  # Width of each frame
batch_size = 8
num_ratings = 5  # Predict ratings for 5 attributes: shots hit, shots missed, shots fired, crosshair placement, accuracy

# Initialize the model
model = VideoRatingModel(num_ratings, input_frames, input_height, input_width).to(device)

# Load dataset
video_dataset = VideoDataset(video_folder='Dataset/missed', input_frames=input_frames)
dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True)

# Loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss for regression tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for videos, labels in dataloader:
        videos, labels = videos.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)  # Forward pass
        loss = criterion(outputs, labels)  # MSELoss for ratings
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Save checkpoint
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'video_rating_model_epoch_{epoch+1}.pth')
        print(f"Model checkpoint saved for epoch {epoch+1}")

# Final model save
torch.save(model.state_dict(), 'video_rating_model_final.pth')
print("Final model saved as video_rating_model_final.pth")

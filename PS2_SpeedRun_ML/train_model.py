import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import device
import torch.nn.functional as F

class VideoDataset(Dataset):
    def __init__(self, video_folder, transform=None):
        self.video_folder = video_folder
        self.transform = transform
        self.video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4') or f.endswith('.webm')]
        self.labels = [0 if 'missed' in f else 1 for f in self.video_files]  # Assuming 'missed' indicates missed shots

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

        # Stack frames and convert to tensor
        frames = np.array(frames)

        # Ensure the shape is (num_frames, height, width, channels)
        # Convert to (num_frames, channels, height, width)
        frames = frames.transpose((0, 3, 1, 2))  # Change to (num_frames, 3, height, width)
        frames = torch.tensor(frames).float()  # Convert to float tensor

        # Ensure you select a fixed number of frames (input_frames)
        if frames.size(0) < input_frames:
            # Padding or selecting only the first `input_frames` if there are fewer frames
            frames = F.pad(frames, (0, 0, 0, 0, 0, input_frames - frames.size(0)), "constant", 0)
        else:
            frames = frames[:input_frames]  # Select the first `input_frames`

        # Make sure frames are in shape (batch_size, channels, depth, height, width)
        return frames, self.labels[idx]


# Define a simple 3D CNN model
class VideoClassificationModel(nn.Module):
    def __init__(self, num_classes, input_frames, input_height, input_width):
        super(VideoClassificationModel, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )
        
        # Calculate the output size after convolutions and pooling
        self.flattened_size = self._get_flattened_size(input_frames, input_height, input_width)

        self.fc = nn.Linear(self.flattened_size, num_classes)

    def _get_flattened_size(self, input_frames, input_height, input_width):
        x = torch.zeros(1, 3, input_frames, input_height, input_width)  # Create a dummy input tensor
        x = self.conv3d(x)  # Forward pass through conv layers
        return int(np.prod(x.shape[1:]))  # Get the flattened size

    def forward(self, x):
        x = self.conv3d(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example dimensions
input_frames = 3  # Number of frames you want to feed into the model
input_height = 240  # Height of each frame
input_width = 320  # Width of each frame

# Initialize model
num_classes = 2  # "hit" and "missed"
model = VideoClassificationModel(num_classes, input_frames, input_height, input_width).to(device)

# Load dataset
video_dataset = VideoDataset(video_folder='Dataset/missed')
dataloader = DataLoader(video_dataset, batch_size=1, shuffle=True)

# Loss and optimizer
class_weights = torch.tensor([1.0, 2.0])
criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for videos, labels in dataloader:
        videos, labels = videos.to(device), labels.to(device)  # Move to device

        optimizer.zero_grad()
        outputs = model(videos)  # This should have the shape [batch_size, num_classes]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Save the model
# torch.save(model.state_dict(), 'video_shot_classification_model.pth')
# print("Model saved as video_shot_classification_model.pth")

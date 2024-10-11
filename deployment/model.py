# deployment/model.py

import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from .models import CNN_LSTM
import traceback  # For detailed error logging


# Define the FrameTransform class
class FrameTransform:
    def __init__(self):
        mean = [0.47670888900756836, 0.4601706266403198, 0.43789467215538025]
        std = [0.19997993111610413, 0.22317750751972198, 0.26677823066711426]
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, frames):
        C, T, H, W = frames.shape
        frames = frames.permute(1, 0, 2, 3)
        normalized_frames = []
        for t in range(T):
            frame = frames[t]
            frame = self.normalize(frame)
            normalized_frames.append(frame)
        frames = torch.stack(normalized_frames)
        frames = frames.permute(1, 0, 2, 3)
        return frames


# Load the trained model
def load_model(model_path):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNN_LSTM(
            num_classes=2,
            hidden_dim=256,
            num_layers=2,
            dropout=0.5
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from {model_path} on {device}")
        return model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        print(traceback.format_exc())
        raise e  # Re-raise the exception to prevent the server from running without a model


# Initialize model at module load
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pth')
model, device = load_model(MODEL_PATH)
transform = FrameTransform()


# Prediction function
def predict(video_path):
    frames_per_video = 16
    frames = []

    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames >= frames_per_video:
            frame_indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)
        else:
            frame_indices = np.linspace(0, total_frames - 1, frames_per_video, dtype=int)

        current_frame = 0
        sampled_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                sampled_frames.append(frame)
            current_frame += 1
            if len(sampled_frames) == frames_per_video:
                break
        cap.release()

        # If not enough frames, duplicate the last frame
        while len(sampled_frames) < frames_per_video:
            sampled_frames.append(sampled_frames[-1])

        frames = np.stack(sampled_frames)
        frames = frames.transpose((3, 0, 1, 2))  # (C, T, H, W)
        frames = torch.FloatTensor(frames) / 255.0  # Normalize to [0,1]
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        print(traceback.format_exc())
        frames = torch.zeros((3, frames_per_video, 224, 224))

    try:
        frames = transform(frames)
        frames = frames.unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            outputs = model(frames)
            _, preds = torch.max(outputs, 1)
            prediction = preds.item()

        label_mapping = {0: 'Non-Shoplifting', 1: 'Shoplifting'}
        return label_mapping.get(prediction, "Unknown")
    except Exception as e:
        print(f"Error during prediction: {e}")
        print(traceback.format_exc())
        return "Error during prediction."

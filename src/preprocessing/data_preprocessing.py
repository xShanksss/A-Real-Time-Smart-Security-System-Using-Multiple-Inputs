"""
violence_detection_system/src/preprocessing/video_processor.py
Video preprocessing and frame extraction
"""

import cv2
import numpy as np
from pathlib import Path
import json
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

class VideoProcessor:
    def __init__(self, config_path='config/config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.image_size = tuple(self.config['model']['image_size'])
        self.sequence_length = self.config['model']['sequence_length']
        
    def extract_frames(self, video_path, max_frames=None):
        """Extract frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            frame_interval = max(1, total_frames // max_frames)
        else:
            frame_interval = 1
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Resize and preprocess
                frame = cv2.resize(frame, self.image_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        return np.array(frames)
    
    def create_sequences(self, frames):
        """Create sequences for LSTM"""
        sequences = []
        
        if len(frames) < self.sequence_length:
            # Pad if too short
            padding = np.zeros((self.sequence_length - len(frames), 
                              *self.image_size, 3))
            frames = np.vstack([frames, padding])
        
        for i in range(0, len(frames) - self.sequence_length + 1, 
                      self.sequence_length // 2):
            seq = frames[i:i + self.sequence_length]
            sequences.append(seq)
        
        return np.array(sequences)
    
    def preprocess_for_model(self, frames):
        """Preprocess frames for model input"""
        processed = np.array([img_to_array(frame) for frame in frames])
        processed = preprocess_input(processed)
        return processed
    
    def process_video_for_training(self, video_path, label):
        """Process video for training"""
        frames = self.extract_frames(video_path, max_frames=120)
        sequences = self.create_sequences(frames)
        processed = self.preprocess_for_model(sequences.reshape(-1, *self.image_size, 3))
        processed = processed.reshape(-1, self.sequence_length, *self.image_size, 3)
        
        labels = np.array([label] * len(processed))
        return processed, labels
    
    def process_dataset(self, dataset_path, output_path):
        """Process entire dataset"""
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        
        X_train, y_train = [], []
        X_test, y_test = [], []
        
        # Process training data
        print("Processing training data...")
        for label_idx, label in enumerate(['non_violence', 'violence']):
            label_path = dataset_path / 'train' / label
            if label_path.exists():
                for video_file in label_path.glob('*.mp4'):
                    try:
                        X, y = self.process_video_for_training(video_file, label_idx)
                        X_train.append(X)
                        y_train.append(y)
                        print(f"✓ Processed: {video_file.name}")
                    except Exception as e:
                        print(f"✗ Error processing {video_file.name}: {e}")
        
        # Process test data
        print("\nProcessing test data...")
        for label_idx, label in enumerate(['non_violence', 'violence']):
            label_path = dataset_path / 'test' / label
            if label_path.exists():
                for video_file in label_path.glob('*.mp4'):
                    try:
                        X, y = self.process_video_for_training(video_file, label_idx)
                        X_test.append(X)
                        y_test.append(y)
                        print(f"✓ Processed: {video_file.name}")
                    except Exception as e:
                        print(f"✗ Error processing {video_file.name}: {e}")
        
        # Combine and save
        if X_train:
            X_train = np.vstack(X_train)
            y_train = np.hstack(y_train)
            np.save(output_path / 'X_train.npy', X_train)
            np.save(output_path / 'y_train.npy', y_train)
        
        if X_test:
            X_test = np.vstack(X_test)
            y_test = np.hstack(y_test)
            np.save(output_path / 'X_test.npy', X_test)
            np.save(output_path / 'y_test.npy', y_test)
        
        print(f"\n✓ Dataset processed and saved to {output_path}")
        return X_train.shape, X_test.shape

if __name__ == "__main__":
    processor = VideoProcessor()
    dataset_dir = Path("datasets")
    processor.process_dataset(dataset_dir, dataset_dir)

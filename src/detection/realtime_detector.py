"""
violence_detection_system/src/detection/realtime_detector.py
Real-time violence detection from video streams
"""

import cv2
import numpy as np
from collections import deque
import json
from pathlib import Path
import time
from datetime import datetime
import tensorflow as tf

class RealtimeViolenceDetector:
    def __init__(self, model_path='models/trained/violence_detector.h5',
                 config_path='config/config.json'):
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        print("✓ Model loaded successfully")
        
        # Configuration
        self.sequence_length = self.config['model']['sequence_length']
        self.image_size = tuple(self.config['model']['image_size'])
        self.confidence_threshold = self.config['detection']['confidence_threshold']
        self.violence_threshold = self.config['detection']['violence_threshold']
        self.alert_cooldown = self.config['detection']['alert_cooldown']
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=self.sequence_length)
        
        # Detection state
        self.last_alert_time = 0
        self.detection_log = []
        
    def preprocess_frame(self, frame):
        """Preprocess single frame"""
        frame = cv2.resize(frame, self.image_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype('float32')
        frame = tf.keras.applications.resnet50.preprocess_input(frame)
        return frame
    
    def predict_sequence(self):
        """Predict violence from current frame buffer"""
        if len(self.frame_buffer) < self.sequence_length:
            return 0.0
        
        # Prepare sequence
        sequence = np.array(list(self.frame_buffer))
        sequence = np.expand_dims(sequence, axis=0)
        
        # Predict
        prediction = self.model.predict(sequence, verbose=0)[0][0]
        return float(prediction)
    
    def detect_from_video(self, video_path, output_path=None, 
                         callback=None, show_video=True):
        """
        Detect violence from video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            callback: Function to call on violence detection
            show_video: Whether to display video
        """
        
        cap = cv2.VideoCapture(str(video_path))
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")
        
        frame_count = 0
        violence_detections = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess and add to buffer
            processed_frame = self.preprocess_frame(frame)
            self.frame_buffer.append(processed_frame)
            
            # Predict every few frames
            if frame_count % 5 == 0 and len(self.frame_buffer) == self.sequence_length:
                prediction = self.predict_sequence()
                
                # Detect violence
                is_violence = prediction >= self.violence_threshold
                current_time = time.time()
                
                if is_violence:
                    violence_detections.append({
                        'frame': frame_count,
                        'time': frame_count / fps,
                        'confidence': prediction
                    })
                    
                    # Check cooldown
                    if current_time - self.last_alert_time >= self.alert_cooldown:
                        self.last_alert_time = current_time
                        
                        # Trigger callback
                        if callback:
                            callback({
                                 "frame": frame_count,
                                 "confidence": prediction,
                                 "location": str(video_path),
                                 "screenshot_path": None
                            })
                        
                        print(f"⚠️  VIOLENCE DETECTED at frame {frame_count} "
                              f"(Time: {frame_count/fps:.2f}s, "
                              f"Confidence: {prediction:.2%})")
                
                # Draw on frame
                color = (0, 0, 255) if is_violence else (0, 255, 0)
                label = f"{'VIOLENCE' if is_violence else 'SAFE'}: {prediction:.2%}"
                
                cv2.rectangle(frame, (10, 10), (400, 60), (0, 0, 0), -1)
                cv2.putText(frame, label, (20, 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Write frame
            if writer:
                writer.write(frame)
            
            # Display frame
            if show_video:
                cv2.imshow('Violence Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
            # Progress
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show_video:
            cv2.destroyAllWindows()
        
        # Generate report
        report = {
            'video_path': str(video_path),
            'total_frames': total_frames,
            'duration': total_frames / fps,
            'fps': fps,
            'violence_detections': len(violence_detections),
            'detections': violence_detections,
            'processed_at': datetime.now().isoformat()
        }
        
        # Save report
        report_path = Path('results/detections') / f"report_{int(time.time())}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"\n✓ Processing complete!")
        print(f"✓ Total violence detections: {len(violence_detections)}")
        print(f"✓ Report saved to: {report_path}")
        
        return report
    
    def detect_from_webcam(self, callback=None):
        """Detect violence from webcam stream"""
        
        cap = cv2.VideoCapture(0)
        print("Starting webcam detection... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess and add to buffer
            processed_frame = self.preprocess_frame(frame)
            self.frame_buffer.append(processed_frame)
            
            # Predict
            if len(self.frame_buffer) == self.sequence_length:
                prediction = self.predict_sequence()
                is_violence = prediction >= self.violence_threshold
                current_time = time.time()
                
                if is_violence and current_time - self.last_alert_time >= self.alert_cooldown:
                    self.last_alert_time = current_time
                    if callback:
                        callback({
                           "frame": -1,
                           "confidence": prediction,
                           "location": "Webcam",
                           "screenshot_path": None
                        })
                    print(f"⚠️  VIOLENCE DETECTED! Confidence: {prediction:.2%}")
                
                # Draw on frame
                color = (0, 0, 255) if is_violence else (0, 255, 0)
                label = f"{'VIOLENCE' if is_violence else 'SAFE'}: {prediction:.2%}"
                
                cv2.rectangle(frame, (10, 10), (400, 60), (0, 0, 0), -1)
                cv2.putText(frame, label, (20, 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow('Violence Detection - Webcam', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Test function
def test_detector():
    """Test the detector on a video"""
    detector = RealtimeViolenceDetector()
    
    # Test on video
    video_path = 'datasets/raw_videos/test_video.mp4'
    output_path = 'results/detections/output_video.mp4'
    
    def alert_callback(data):
        print(f"🚨 ALERT! Violence detected with {data['confidence']:.2%} confidence")
    
    report = detector.detect_from_video(
        video_path, 
        output_path=output_path,
        callback=alert_callback,
        show_video=True
    )
    
    return report

if __name__ == "__main__":
    test_detector()

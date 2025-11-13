"""
Real-Time Smart Security System - Violence Detection
Project Structure Setup Script
"""

import os
import json

def create_project_structure():
    """Create complete project folder structure"""
    
    folders = [
        'violence_detection_system',
        'violence_detection_system/models',
        'violence_detection_system/models/trained',
        'violence_detection_system/datasets',
        'violence_detection_system/datasets/train',
        'violence_detection_system/datasets/train/violence',
        'violence_detection_system/datasets/train/non_violence',
        'violence_detection_system/datasets/test',
        'violence_detection_system/datasets/test/violence',
        'violence_detection_system/datasets/test/non_violence',
        'violence_detection_system/datasets/raw_videos',
        'violence_detection_system/src',
        'violence_detection_system/src/preprocessing',
        'violence_detection_system/src/models',
        'violence_detection_system/src/detection',
        'violence_detection_system/src/alerts',
        'violence_detection_system/web',
        'violence_detection_system/web/static',
        'violence_detection_system/web/static/css',
        'violence_detection_system/web/static/js',
        'violence_detection_system/web/templates',
        'violence_detection_system/web/uploads',
        'violence_detection_system/results',
        'violence_detection_system/results/detections',
        'violence_detection_system/results/logs',
        'violence_detection_system/config',
        'violence_detection_system/notebooks',
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✓ Created: {folder}")
    
    # Create configuration file
    config = {
        "model": {
            "type": "hybrid",
            "cnn_model": "ResNet50",
            "lstm_units": 128,
            "sequence_length": 16,
            "image_size": [224, 224],
            "batch_size": 32,
            "epochs": 50
        },
        "detection": {
            "confidence_threshold": 0.75,
            "violence_threshold": 0.8,
            "alert_cooldown": 30
        },
        "alerts": {
            "sms_enabled": True,
            "email_enabled": True,
            "sound_enabled": True
        },
        "video": {
            "fps": 30,
            "max_duration": 3600
        }
    }
    
    with open('violence_detection_system/config/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\n✓ Configuration file created")
    print("\n" + "="*60)
    print("Project structure created successfully!")
    print("="*60)

if __name__ == "__main__":
    create_project_structure()
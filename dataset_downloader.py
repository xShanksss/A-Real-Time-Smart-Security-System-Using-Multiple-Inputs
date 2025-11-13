"""
violence_detection_system/download_datasets.py
Download and prepare datasets for violence detection
"""

import os
import urllib.request
import zipfile
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

class DatasetDownloader:
    def __init__(self):
        self.base_path = Path('datasets')
        self.base_path.mkdir(exist_ok=True)
        
    def download_file(self, url, destination):
        """Download file with progress bar"""
        
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)
        
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                                desc=Path(destination).name) as t:
            urllib.request.urlretrieve(url, filename=destination, 
                                     reporthook=t.update_to)
    
    def create_sample_videos(self):
        """Create sample videos for testing when real datasets are not available"""
        
        print("\n📹 Creating sample videos for testing...")
        
        def create_video(path, label, num_frames=120):
            """Create a synthetic video"""
            
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(path), fourcc, 30.0, (640, 480))
            
            for i in range(num_frames):
                # Create frame based on label
                if label == 'violence':
                    # Red-tinted frames with motion
                    frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
                    frame[:, :, 0] = 50  # Less blue
                    frame[:, :, 1] = 50  # Less green
                    
                    # Add "motion" artifacts
                    if i % 10 < 5:
                        frame = cv2.GaussianBlur(frame, (15, 15), 0)
                else:
                    # Normal frames
                    frame = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
                    
                    # Add some structure
                    cv2.rectangle(frame, (50, 50), (590, 430), (200, 200, 200), 2)
                
                # Add text
                text = f"{'VIOLENCE' if label == 'violence' else 'NORMAL'} - Frame {i}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 255, 255), 2)
                
                out.write(frame)
            
            out.release()
            print(f"✓ Created: {path}")
        
        # Create training videos
        train_violence = self.base_path / 'train' / 'violence'
        train_normal = self.base_path / 'train' / 'non_violence'
        
        for i in range(10):
            create_video(train_violence / f'violence_{i}.mp4', 'violence', 150)
            create_video(train_normal / f'normal_{i}.mp4', 'non_violence', 150)
        
        # Create test videos
        test_violence = self.base_path / 'test' / 'violence'
        test_normal = self.base_path / 'test' / 'non_violence'
        
        for i in range(3):
            create_video(test_violence / f'test_violence_{i}.mp4', 'violence', 120)
            create_video(test_normal / f'test_normal_{i}.mp4', 'non_violence', 120)
        
        # Create raw test videos
        raw_videos = self.base_path / 'raw_videos'
        raw_videos.mkdir(exist_ok=True)
        create_video(raw_videos / 'test_video.mp4', 'violence', 200)
        create_video(raw_videos / 'demo_video.mp4', 'non_violence', 200)
        
        print("✓ Sample videos created successfully!")
    
    def download_real_datasets(self):
        """
        Instructions for downloading real datasets
        Note: These require manual download due to licensing/access restrictions
        """
        
        print("\n" + "="*70)
        print("REAL DATASET DOWNLOAD INSTRUCTIONS")
        print("="*70)
        
        datasets = [
            {
                'name': 'RWF-2000 (Real World Fight)',
                'description': '2000 videos of fights and non-fights',
                'url': 'https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection',
                'size': '~2GB',
                'accuracy_potential': '85-90%'
            },
            {
                'name': 'UCF Crime Dataset',
                'description': 'Large-scale anomaly detection dataset',
                'url': 'https://www.crcv.ucf.edu/projects/real-world/',
                'size': '~10GB',
                'accuracy_potential': '80-88%'
            },
            {
                'name': 'Violent Flows',
                'description': 'Crowd violence detection',
                'url': 'http://www.openu.ac.il/home/hassner/data/violentflows/',
                'size': '~500MB',
                'accuracy_potential': '82-87%'
            },
            {
                'name': 'Hockey Fight Dataset',
                'description': 'Hockey fight videos',
                'url': 'https://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89',
                'size': '~1GB',
                'accuracy_potential': '88-92%'
            }
        ]
        
        for i, dataset in enumerate(datasets, 1):
            print(f"\n{i}. {dataset['name']}")
            print(f"   Description: {dataset['description']}")
            print(f"   URL: {dataset['url']}")
            print(f"   Size: {dataset['size']}")
            print(f"   Expected Accuracy: {dataset['accuracy_potential']}")
        
        print("\n" + "="*70)
        print("DOWNLOAD STEPS:")
        print("="*70)
        print("1. Visit the dataset URLs above")
        print("2. Follow their specific download instructions")
        print("3. Extract videos to:")
        print(f"   - Training: {self.base_path / 'train' / '[violence|non_violence]'}")
        print(f"   - Testing: {self.base_path / 'test' / '[violence|non_violence]'}")
        print("4. Organize videos:")
        print("   datasets/")
        print("   ├── train/")
        print("   │   ├── violence/       (violence videos)")
        print("   │   └── non_violence/   (normal videos)")
        print("   └── test/")
        print("       ├── violence/       (violence test videos)")
        print("       └── non_violence/   (normal test videos)")
        print("\n" + "="*70)
    
    def verify_dataset(self):
        """Verify dataset structure and count files"""
        
        print("\n📊 Verifying dataset...")
        
        paths = [
            self.base_path / 'train' / 'violence',
            self.base_path / 'train' / 'non_violence',
            self.base_path / 'test' / 'violence',
            self.base_path / 'test' / 'non_violence'
        ]
        
        total_videos = 0
        for path in paths:
            if path.exists():
                count = len(list(path.glob('*.mp4'))) + len(list(path.glob('*.avi')))
                total_videos += count
                print(f"✓ {path.relative_to(self.base_path)}: {count} videos")
            else:
                print(f"✗ {path.relative_to(self.base_path)}: Not found")
        
        print(f"\nTotal videos: {total_videos}")
        
        if total_videos >= 20:
            print("✓ Dataset ready for training!")
            return True
        else:
            print("⚠️  Insufficient videos. Minimum 20 videos recommended.")
            return False
    
    def create_readme(self):
        """Create README for dataset"""
        
        readme_content = """# Violence Detection Dataset

## Structure

```
datasets/
├── train/
│   ├── violence/       # Training videos with violence
│   └── non_violence/   # Training videos without violence
├── test/
│   ├── violence/       # Test videos with violence
│   └── non_violence/   # Test videos without violence
├── raw_videos/         # Raw videos for testing
├── X_train.npy        # Processed training features
├── y_train.npy        # Training labels
├── X_test.npy         # Processed test features
└── y_test.npy         # Test labels

## Video Format

- Supported formats: MP4, AVI, MOV, MKV
- Recommended resolution: 640x480 or higher
- Recommended duration: 5-30 seconds per clip
- Frame rate: 25-30 FPS

## Data Distribution

For best results:
- Minimum 50 videos per class (violence/non-violence)
- 80-20 train-test split
- Balanced classes (equal videos in each category)

## Labels

- **0**: Non-violence (normal behavior)
- **1**: Violence (fighting, assault, aggressive behavior)

## Sources

Videos can be collected from:
1. Public datasets (RWF-2000, UCF Crime, etc.)
2. CCTV footage (with proper permissions)
3. YouTube videos (following fair use guidelines)

## Processing

Videos are processed into sequences of frames:
- Frame extraction at regular intervals
- Resize to 224x224 pixels
- Normalized using ResNet50 preprocessing
- Organized into sequences of 16 frames

## Privacy & Ethics

- Ensure proper consent for any real footage
- Anonymize faces and identifiable information
- Use only for security and research purposes
- Follow local laws and regulations
"""
        
        readme_path = self.base_path / 'README.md'
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"✓ Created: {readme_path}")

def main():
    """Main function"""
    
    print("="*70)
    print("VIOLENCE DETECTION DATASET SETUP")
    print("="*70)
    
    downloader = DatasetDownloader()
    
    # Create directory structure
    print("\n📁 Creating directory structure...")
    downloader.base_path.mkdir(exist_ok=True)
    
    # Show real dataset options
    downloader.download_real_datasets()
    
    # Ask user
    print("\n" + "="*70)
    response = input("\nCreate sample videos for testing? (y/n): ").lower()
    
    if response == 'y':
        downloader.create_sample_videos()
    
    # Verify dataset
    downloader.verify_dataset()
    
    # Create README
    downloader.create_readme()
    
    print("\n" + "="*70)
    print("✓ DATASET SETUP COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. If using sample data: Ready to train!")
    print("2. If using real data: Download datasets and organize them")
    print("3. Run: python src/preprocessing/video_processor.py")
    print("4. Run: python src/models/violence_detector.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

# violence_detection_system/requirements.txt
# Python requirements for Violence Detection System

# Deep Learning Frameworks
tensorflow==2.15.0
keras==2.15.0

# Computer Vision
opencv-python==4.8.1.78
opencv-contrib-python==4.8.1.78

# Data Processing
numpy==1.24.3
pandas==2.1.0
scikit-learn==1.3.0
scipy==1.11.2

# Image Processing
Pillow==10.0.0
imageio==2.31.3

# Web Framework
Flask==3.0.0
Flask-CORS==4.0.0
Werkzeug==3.0.0

# Alerts and Notifications
twilio==8.9.0
# For SMS alerts - requires account setup

# Utilities
python-dotenv==1.0.0
tqdm==4.66.1
pathlib==1.0.1

# Visualization
matplotlib==3.8.0
seaborn==0.12.2

# Video Processing
moviepy==1.0.3

# Configuration
pyyaml==6.0.1

# API and HTTP
requests==2.31.0

# Progress bars
colorama==0.4.6

# ============================================
# INSTALLATION INSTRUCTIONS
# ============================================

"""
STEP 1: Create Virtual Environment
-----------------------------------
python -m venv venv

# Activate (Windows)
venv\\Scripts\\activate

# Activate (Linux/Mac)
source venv/bin/activate


STEP 2: Install Requirements
-----------------------------
pip install --upgrade pip
pip install -r requirements.txt


STEP 3: Download Pre-trained Models (Optional)
-----------------------------------------------
# ResNet50 weights will download automatically on first run
# Or download manually:
python -c "from tensorflow.keras.applications import ResNet50; ResNet50(weights='imagenet')"


STEP 4: Configure Alerts
-------------------------
1. For SMS Alerts (Twilio):
   - Sign up at https://www.twilio.com/
   - Get Account SID and Auth Token
   - Update config/alert_config.json

2. For Email Alerts:
   - Use Gmail with App Password
   - Enable 2FA on Gmail
   - Generate App Password: https://myaccount.google.com/apppasswords
   - Update config/alert_config.json


STEP 5: Prepare Dataset
------------------------
Run the dataset download script:
python download_datasets.py


STEP 6: Train Model
-------------------
# Process videos
python src/preprocessing/video_processor.py

# Train model
python src/models/violence_detector.py


STEP 7: Run Web Application
----------------------------
python web/app.py

Access at: http://localhost:5000


TROUBLESHOOTING
---------------

1. TensorFlow GPU Issues:
   pip install tensorflow[and-cuda]

2. OpenCV Issues:
   pip uninstall opencv-python opencv-contrib-python
   pip install opencv-python==4.8.1.78

3. Windows Sound Issues:
   Already using winsound (built-in)

4. Linux Sound Issues:
   Install: sudo apt-get install beep

5. Memory Issues:
   Reduce batch_size in config/config.json


SYSTEM REQUIREMENTS
-------------------
- Python 3.8 - 3.11
- RAM: 8GB minimum (16GB recommended)
- GPU: NVIDIA GPU with CUDA (optional but recommended)
- Storage: 10GB free space
- OS: Windows 10/11, Ubuntu 20.04+, macOS 10.15+


DATASET SOURCES
---------------
For training, you can use these public datasets:

1. UCF Crime Dataset
   - https://www.crcv.ucf.edu/projects/real-world/

2. Violent Flows Dataset
   - http://www.openu.ac.il/home/hassner/data/violentflows/

3. Hockey Fight Dataset
   - https://academictorrents.com/details/38d9ed996a5a75a039b84cf8a137be794e7cee89

4. RWF-2000 Dataset (Real-World Fight)
   - https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection


PERFORMANCE TIPS
----------------
1. Use GPU for training (10-20x faster)
2. Reduce video resolution for faster processing
3. Adjust sequence_length for memory vs accuracy tradeoff
4. Use batch processing for multiple videos
5. Enable model quantization for deployment
"""

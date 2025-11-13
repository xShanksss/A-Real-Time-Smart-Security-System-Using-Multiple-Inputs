"""
violence_detection_system/web/app.py
Flask web application for violence detection system
"""

from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
import os
import json
import threading
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.detection.realtime_detector import RealtimeViolenceDetector
from src.alerts.alert_system import AlertManager

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'web/uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Global variables
detector = None
alert_manager = None
processing_status = {
    'is_processing': False,
    'progress': 0,
    'current_video': None,
    'detections': [],
    'report': None
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def initialize_system():
    """Initialize detector and alert manager"""
    global detector, alert_manager
    
    try:
        detector = RealtimeViolenceDetector(
            model_path='models/trained/violence_detector.h5',
            config_path='config/config.json'
        )
        alert_manager = AlertManager(config_path='config/alert_config.json')
        print("✓ System initialized successfully")
        return True
    except Exception as e:
        print(f"✗ System initialization failed: {e}")
        return False

@app.route('/')
def index():
    """Home page"""
    return render_template('html_template.html')

@app.route('/api/status')
def get_status():
    """Get processing status"""
    return jsonify(processing_status)

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload video for processing"""
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        filename = f"{timestamp}_{filename}"
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        return jsonify({
            'message': 'Video uploaded successfully',
            'filename': filename,
            'filepath': filepath
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/process', methods=['POST'])
def process_video():
    """Process uploaded video"""
    global processing_status, detector, alert_manager
    
    if processing_status['is_processing']:
        return jsonify({'error': 'Another video is being processed'}), 400
    
    data = request.json
    video_path = data.get('video_path')
    enable_alerts = data.get('enable_alerts', True)
    
    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'Invalid video path'}), 400
    
    # Initialize if needed
    if detector is None:
        if not initialize_system():
            return jsonify({'error': 'System initialization failed'}), 500
    
    # Reset status
    processing_status['is_processing'] = True
    processing_status['progress'] = 0
    processing_status['current_video'] = video_path
    processing_status['detections'] = []
    processing_status['report'] = None
    
    # Alert callback
    def alert_callback(data):
             frame = data['frame']
             confidence = data['confidence']

             detection = {
                 'frame': frame,
                 'confidence': float(confidence),
                 'timestamp': time.time()
             }
             processing_status['detections'].append(detection)
      
             if enable_alerts and alert_manager:
                 data['location'] = f"Video: {Path(video_path).name}"
                 alert_manager.trigger_alert(data)
    
    # Process in thread
    def process_thread():
        try:
            output_path = video_path.replace('.mp4', '_processed.mp4')
            report = detector.detect_from_video(
                video_path,
                output_path=output_path,
                callback=alert_callback,
                show_video=False
            )
            
            processing_status['report'] = report
            processing_status['is_processing'] = False
            processing_status['progress'] = 100
            
        except Exception as e:
            print(f"Processing error: {e}")
            processing_status['is_processing'] = False
            processing_status['error'] = str(e)
    
    thread = threading.Thread(target=process_thread, daemon=True)
    thread.start()
    
    return jsonify({'message': 'Processing started'})

@app.route('/api/detections')
def get_detections():
    """Get detection history"""
    return jsonify(processing_status['detections'])

@app.route('/api/report')
def get_report():
    """Get processing report"""
    if processing_status['report']:
        return jsonify(processing_status['report'])
    return jsonify({'error': 'No report available'}), 404

@app.route('/api/alerts/configure', methods=['POST'])
def configure_alerts():
    """Configure alert settings"""
    global alert_manager
    
    if alert_manager is None:
        initialize_system()
    
    data = request.json
    alert_type = data.get('type')
    config = data.get('config')
    
    if alert_manager.configure_alerts(alert_type, config):
        return jsonify({'message': 'Alert configuration updated'})
    
    return jsonify({'error': 'Configuration failed'}), 400

@app.route('/api/alerts/history')
def get_alert_history():
    """Get alert history"""
    global alert_manager
    
    if alert_manager is None:
        return jsonify([])
    
    limit = request.args.get('limit', 10, type=int)
    history = alert_manager.get_alert_history(limit)
    return jsonify(history)

@app.route('/api/webcam/start', methods=['POST'])
def start_webcam():
    """Start webcam detection"""
    # This would require a different implementation for web-based webcam
    return jsonify({'message': 'Webcam detection not implemented for web interface'})

@app.route('/results/<path:filename>')
def download_result(filename):
    """Download result files"""
    file_path = os.path.join('results', filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('web/uploads', exist_ok=True)
    os.makedirs('results/detections', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)
    
    # Initialize system
    initialize_system()
    
    # Run app
    print("\n" + "="*60)
    print("🚀 Starting Violence Detection Web Application")
    print("="*60)
    print("Access the application at: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

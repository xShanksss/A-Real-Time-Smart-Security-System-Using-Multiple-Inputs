// violence_detection_system/web/static/js/app.js

let currentVideoFile = null;
let currentVideoPath = null;
let statusCheckInterval = null;
let startTime = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    setupUploadZone();
    setupButtons();
    setupVideoPathInput();
});

function setupUploadZone() {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');

    uploadZone.addEventListener('click', () => fileInput.click());

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

function setupButtons() {
    document.getElementById('processBtn').addEventListener('click', startProcessing);
}

function setupVideoPathInput() {
    const videoPathInput = document.getElementById('videoPath');
    videoPathInput.addEventListener('input', (e) => {
        if (e.target.value.trim()) {
            currentVideoPath = e.target.value.trim();
            document.getElementById('processBtn').disabled = false;
        } else {
            document.getElementById('processBtn').disabled = !currentVideoFile;
        }
    });
}

async function handleFileSelect(file) {
    if (!file.type.startsWith('video/')) {
        showAlert('Please select a valid video file', 'danger');
        return;
    }

    currentVideoFile = file;
    currentVideoPath = null;
    
    // Show preview
    const videoPreview = document.getElementById('videoPreview');
    videoPreview.src = URL.createObjectURL(file);
    videoPreview.style.display = 'block';

    // Update UI
    document.getElementById('uploadZone').innerHTML = `
        <p style="font-size: 2em; margin-bottom: 10px;">✅</p>
        <p style="font-size: 1.2em; margin-bottom: 10px;">Video loaded: ${file.name}</p>
        <p style="color: #888;">Size: ${(file.size / (1024*1024)).toFixed(2)} MB</p>
    `;

    document.getElementById('processBtn').disabled = false;

    // Upload file
    await uploadFile(file);
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('video', file);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (response.ok) {
            currentVideoPath = data.filepath;
            console.log('Video uploaded:', data.filename);
        } else {
            showAlert(data.error, 'danger');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showAlert('Upload failed', 'danger');
    }
}

async function startProcessing() {
    if (!currentVideoPath) {
        showAlert('No video selected', 'danger');
        return;
    }

    const processBtn = document.getElementById('processBtn');
    processBtn.disabled = true;
    processBtn.querySelector('span').innerHTML = '⏳ Processing...';

    // Show progress section
    document.getElementById('progressSection').style.display = 'block';

    // Get alert settings
    const enableAlerts = document.getElementById('enableSMS').checked ||
                        document.getElementById('enableEmail').checked ||
                        document.getElementById('enableSound').checked;

    startTime = Date.now();

    try {
        const response = await fetch('/api/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                video_path: currentVideoPath,
                enable_alerts: enableAlerts
            })
        });

        const data = await response.json();

        if (response.ok) {
            console.log('Processing started');
            startStatusCheck();
        } else {
            showAlert(data.error, 'danger');
            resetUI();
        }
    } catch (error) {
        console.error('Processing error:', error);
        showAlert('Processing failed', 'danger');
        resetUI();
    }
}

function startStatusCheck() {
    statusCheckInterval = setInterval(checkStatus, 1000);
}

async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();

        // Update progress bar
        const progressBar = document.getElementById('progressBar');
        progressBar.style.width = status.progress + '%';
        progressBar.textContent = status.progress + '%';

        // Update status text
        const statusText = document.getElementById('statusText');
        if (status.is_processing) {
            statusText.textContent = 'Processing video... Please wait';
        } else {
            statusText.textContent = 'Processing complete!';
        }

        // Update detections
        if (status.detections.length > 0) {
            updateDetectionsList(status.detections);
            updateStatistics(status);
        }

        // Check if done
        if (!status.is_processing && status.progress === 100) {
            clearInterval(statusCheckInterval);
            processingComplete(status);
        }
    } catch (error) {
        console.error('Status check error:', error);
    }
}

function updateDetectionsList(detections) {
    const detectionsList = document.getElementById('detectionsList');
    
    const html = detections.map((det, index) => {
        const time = det.frame > 0 ? `Frame ${det.frame}` : 'Live';
        const confidence = (det.confidence * 100).toFixed(1);
        
        return `
            <div class="detection-item danger">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>🚨 Violence Detection #${index + 1}</strong>
                        <p style="margin-top: 5px; color: #666;">${time}</p>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.5em; font-weight: bold; color: #f44336;">
                            ${confidence}%
                        </div>
                        <small style="color: #666;">Confidence</small>
                    </div>
                </div>
            </div>
        `;
    }).join('');

    detectionsList.innerHTML = html;

    // Show alert banner for latest detection
    const latest = detections[detections.length - 1];
    showViolenceAlert(latest);
}

function showViolenceAlert(detection) {
    const banner = document.getElementById('alertBanner');
    const message = document.getElementById('alertMessage');
    
    const confidence = (detection.confidence * 100).toFixed(1);
    message.textContent = `Detected at ${detection.frame > 0 ? 'frame ' + detection.frame : 'live stream'} with ${confidence}% confidence`;
    
    banner.classList.add('show');

    // Play alert sound
    if (document.getElementById('enableSound').checked) {
        playAlertSound();
    }

    // Hide after 5 seconds
    setTimeout(() => {
        banner.classList.remove('show');
    }, 5000);
}

function playAlertSound() {
    // Create and play alert sound
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.value = 800;
    oscillator.type = 'sine';
    
    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.5);
}

function updateStatistics(status) {
    const totalDetections = status.detections.length;
    document.getElementById('totalDetections').textContent = totalDetections;

    if (totalDetections > 0) {
        const avgConf = status.detections.reduce((sum, d) => sum + d.confidence, 0) / totalDetections;
        document.getElementById('avgConfidence').textContent = (avgConf * 100).toFixed(1) + '%';
    }

    if (startTime) {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        document.getElementById('processingTime').textContent = elapsed + 's';
    }
}

async function processingComplete(status) {
    console.log('Processing complete:', status);

    // Get final report
    try {
        const response = await fetch('/api/report');
        const report = await response.json();
        
        console.log('Final report:', report);
        
        // Update statistics
        document.getElementById('totalDetections').textContent = report.violence_detections || 0;
        
        const duration = Math.floor(report.duration || 0);
        document.getElementById('processingTime').textContent = duration + 's';
        
    } catch (error) {
        console.error('Report fetch error:', error);
    }

    // Update alert history
    loadAlertHistory();

    // Reset button
    resetUI();
    
    showAlert('Processing complete! Check the results below.', 'success');
}

async function loadAlertHistory() {
    try {
        const response = await fetch('/api/alerts/history?limit=5');
        const history = await response.json();

        if (history.length > 0) {
            const alertHistory = document.getElementById('alertHistory');
            alertHistory.innerHTML = history.map(alert => {
                const time = new Date(alert.timestamp).toLocaleString();
                const confidence = (alert.confidence * 100).toFixed(1);
                
                return `
                    <div class="detection-item">
                        <strong>🔔 Alert: ${alert.type}</strong>
                        <p style="margin-top: 5px; font-size: 0.9em; color: #666;">
                            ${time} • Confidence: ${confidence}%
                        </p>
                    </div>
                `;
            }).join('');
        }
    } catch (error) {
        console.error('Alert history error:', error);
    }
}

function resetUI() {
    const processBtn = document.getElementById('processBtn');
    processBtn.disabled = false;
    processBtn.querySelector('span').innerHTML = '▶️ Start Detection';
}

function showAlert(message, type = 'info') {
    console.log(`${type.toUpperCase()}: ${message}`);
    // You can implement a toast notification here
}

// Auto-refresh alert history
setInterval(loadAlertHistory, 10000);

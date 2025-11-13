"""
violence_detection_system/src/alerts/alert_manager.py
Alert system for violence detection notifications
"""

import json
import time
from datetime import datetime
from pathlib import Path
import threading
import winsound  # For Windows sound alerts
import platform

# For SMS (Twilio)
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("⚠️  Twilio not installed. SMS alerts disabled.")

# For Email
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

class AlertManager:
    def __init__(self, config_path='config/alert_config.json'):
        """Initialize alert manager"""
        
        # Load configuration
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            # Default configuration
            self.config = {
                'sms': {
                    'enabled': False,
                    'twilio_account_sid': 'YOUR_ACCOUNT_SID',
                    'twilio_auth_token': 'YOUR_AUTH_TOKEN',
                    'twilio_phone_number': '+1234567890',
                    'recipient_numbers': ['+1234567890']
                },
                'email': {
                    'enabled': True,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'sender_email': 'your_email@gmail.com',
                    'sender_password': 'your_app_password',
                    'recipient_emails': ['recipient@example.com']
                },
                'sound': {
                    'enabled': True,
                    'alert_sound': 'SystemExclamation'
                },
                'logging': {
                    'enabled': True,
                    'log_file': 'results/logs/alerts.log'
                }
            }
            
            # Save default config
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        
        # Initialize Twilio client
        self.twilio_client = None
        if TWILIO_AVAILABLE and self.config['sms']['enabled']:
            try:
                self.twilio_client = Client(
                    self.config['sms']['twilio_account_sid'],
                    self.config['sms']['twilio_auth_token']
                )
            except Exception as e:
                print(f"⚠️  Twilio initialization failed: {e}")
        
        # Alert log
        self.alert_history = []
    
    def send_sms_alert(self, message):
        """Send SMS alert via Twilio"""
        
        if not self.config['sms']['enabled']:
            return False
        
        if not self.twilio_client:
            print("⚠️  SMS not configured properly")
            return False
        
        success_count = 0
        for recipient in self.config['sms']['recipient_numbers']:
            try:
                message_obj = self.twilio_client.messages.create(
                    body=message,
                    from_=self.config['sms']['twilio_phone_number'],
                    to=recipient
                )
                print(f"✓ SMS sent to {recipient}: {message_obj.sid}")
                success_count += 1
            except Exception as e:
                print(f"✗ Failed to send SMS to {recipient}: {e}")
        
        return success_count > 0
    
    def send_email_alert(self, subject, body, image_path=None):
        """Send email alert"""
        
        if not self.config['email']['enabled']:
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config['email']['sender_email']
            msg['To'] = ', '.join(self.config['email']['recipient_emails'])
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'html'))
            
            # Add image if provided
            if image_path and Path(image_path).exists():
                with open(image_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-Disposition', 'attachment', 
                                 filename=Path(image_path).name)
                    msg.attach(img)
            
            # Send email
            with smtplib.SMTP(self.config['email']['smtp_server'], 
                            self.config['email']['smtp_port']) as server:
                server.starttls()
                server.login(self.config['email']['sender_email'],
                           self.config['email']['sender_password'])
                server.send_message(msg)
            
            print(f"✓ Email sent to {len(self.config['email']['recipient_emails'])} recipients")
            return True
            
        except Exception as e:
            print(f"✗ Failed to send email: {e}")
            return False
    
    def play_sound_alert(self):
        """Play sound alert"""
        
        if not self.config['sound']['enabled']:
            return
        
        def play_sound():
            try:
                if platform.system() == 'Windows':
                    # Windows sound
                    for _ in range(3):
                        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                        time.sleep(0.5)
                else:
                    # Linux/Mac - using system bell
                    import os
                    for _ in range(3):
                        os.system('printf "\a"')
                        time.sleep(0.5)
            except Exception as e:
                print(f"⚠️  Sound alert failed: {e}")
        
        # Play in separate thread to not block
        threading.Thread(target=play_sound, daemon=True).start()
    
    def log_alert(self, alert_data):
        """Log alert to file"""
        
        if not self.config['logging']['enabled']:
            return
        
        log_file = Path(self.config['logging']['log_file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(alert_data) + '\n')
        
        self.alert_history.append(alert_data)
    
    def trigger_alert(self, detection_data):
        """Trigger all enabled alerts"""
        
        timestamp = datetime.now().isoformat()
        frame = detection_data.get('frame', -1)
        confidence = detection_data.get('confidence', 0.0)
        location = detection_data.get('location', 'Unknown')
        
        # Create alert message
        sms_message = (
            f"🚨 VIOLENCE DETECTED!\n"
            f"Time: {timestamp}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Location: {location}"
        )
        
        email_subject = "🚨 Violence Alert - Immediate Action Required"
        email_body = f"""
        <html>
        <body>
            <h2 style="color: red;">⚠️ VIOLENCE DETECTION ALERT</h2>
            <p><strong>Timestamp:</strong> {timestamp}</p>
            <p><strong>Detection Confidence:</strong> {confidence:.1%}</p>
            <p><strong>Frame Number:</strong> {frame}</p>
            <p><strong>Location:</strong> {location}</p>
            <hr>
            <p>Immediate investigation required. Review the attached footage.</p>
            <p style="color: gray; font-size: 12px;">
                This is an automated alert from the Smart Security System.
            </p>
        </body>
        </html>
        """
        
        # Prepare alert data
        alert_data = {
            'timestamp': timestamp,
            'type': 'violence_detection',
            'confidence': confidence,
            'frame': frame,
            'location': location,
            'alerts_sent': {
                'sms': False,
                'email': False,
                'sound': False
            }
        }
        
        print("\n" + "="*60)
        print("🚨 TRIGGERING ALERTS")
        print("="*60)
        
        # Send SMS
        if self.config['sms']['enabled']:
            alert_data['alerts_sent']['sms'] = self.send_sms_alert(sms_message)
        
        # Send Email
        if self.config['email']['enabled']:
            alert_data['alerts_sent']['email'] = self.send_email_alert(
                email_subject, 
                email_body,
                detection_data.get('screenshot_path')
            )
        
        # Play Sound
        if self.config['sound']['enabled']:
            self.play_sound_alert()
            alert_data['alerts_sent']['sound'] = True
        
        # Log alert
        self.log_alert(alert_data)
        
        print("="*60 + "\n")
        
        return alert_data
    
    def get_alert_history(self, limit=10):
        """Get recent alert history"""
        return self.alert_history[-limit:]
    
    def configure_alerts(self, alert_type, config_dict):
        """Update alert configuration"""
        
        if alert_type in self.config:
            self.config[alert_type].update(config_dict)
            
            # Save updated config
            config_file = Path('config/alert_config.json')
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            print(f"✓ {alert_type.upper()} configuration updated")
            return True
        
        return False

# Test function
def test_alerts():
    """Test alert system"""
    
    manager = AlertManager()
    
    # Test alert
    test_detection = {
        'frame': 1234,
        'confidence': 0.95,
        'location': 'Main Entrance - Camera 1'
    }
    
    manager.trigger_alert(test_detection)
    
    print("\n✓ Alert test completed")
    print(f"Alert history: {len(manager.alert_history)} alerts")

if __name__ == "__main__":
    test_alerts()

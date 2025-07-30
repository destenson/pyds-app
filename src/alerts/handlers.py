"""
Alert output handlers with comprehensive delivery mechanisms.

This module provides concrete alert handlers for console, file, webhook, and email
delivery with error handling, retry mechanisms, and custom handler registration.
"""

import asyncio
import time
import json
import smtplib
import aiohttp
import aiofiles
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import ssl
import logging
import sys
import os
from urllib.parse import urlparse
import tempfile
import threading

from ..config import AppConfig, AlertConfig, AlertLevel
from ..utils.errors import AlertError, handle_error
from ..utils.logging import get_logger, performance_context
from .manager import AlertHandler, AlertMessage


class ConsoleHandler(AlertHandler):
    """Handler for console/terminal alert output."""
    
    def __init__(self, name: str = "console", config: Optional[Dict[str, Any]] = None):
        """
        Initialize console handler.
        
        Args:
            name: Handler name
            config: Handler configuration
        """
        config = config or {}
        super().__init__(name, config)
        
        # Console-specific configuration
        self.colored_output = config.get('colored_output', True)
        self.include_timestamp = config.get('include_timestamp', True)
        self.include_metadata = config.get('include_metadata', True)
        self.output_stream = config.get('output_stream', 'stdout')  # stdout or stderr
        self.format_template = config.get('format_template', None)
        
        # Color codes for different alert levels
        self.colors = {
            AlertLevel.LOW: '\033[36m',      # Cyan
            AlertLevel.MEDIUM: '\033[33m',   # Yellow
            AlertLevel.HIGH: '\033[31m',     # Red
            AlertLevel.CRITICAL: '\033[91m', # Bright red
        }
        self.reset_color = '\033[0m'
        
        # Get output stream
        self.stream = sys.stdout if self.output_stream == 'stdout' else sys.stderr
    
    async def _deliver_alert(self, alert: AlertMessage) -> bool:
        """Deliver alert to console."""
        try:
            # Format the alert message
            formatted_message = self._format_alert_message(alert)
            
            # Write to console (thread-safe)
            await self._write_to_console(formatted_message)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Console handler error: {e}")
            return False
    
    def _format_alert_message(self, alert: AlertMessage) -> str:
        """Format alert message for console output."""
        if self.format_template:
            # Use custom template
            return self.format_template.format(
                alert_id=alert.alert_id,
                level=alert.alert_level.value,
                message=alert.message,
                source_id=alert.detection.source_id,
                pattern_name=alert.detection.pattern_name,
                confidence=alert.detection.confidence,
                timestamp=alert.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                **alert.metadata
            )
        
        # Default formatting
        parts = []
        
        # Add timestamp
        if self.include_timestamp:
            timestamp = alert.created_at.strftime('%Y-%m-%d %H:%M:%S')
            parts.append(f"[{timestamp}]")
        
        # Add alert level with color
        level_str = alert.alert_level.value.upper()
        if self.colored_output:
            color = self.colors.get(alert.alert_level, '')
            level_str = f"{color}{level_str}{self.reset_color}"
        parts.append(f"[{level_str}]")
        
        # Add source and pattern
        parts.append(f"[{alert.detection.source_id}:{alert.detection.pattern_name}]")
        
        # Add main message
        parts.append(alert.message)
        
        # Add metadata if requested
        if self.include_metadata and alert.metadata:
            metadata_str = ", ".join([f"{k}={v}" for k, v in alert.metadata.items()])
            parts.append(f"({metadata_str})")
        
        return " ".join(parts)
    
    async def _write_to_console(self, message: str):
        """Write message to console in a thread-safe manner."""
        def _write():
            try:
                print(message, file=self.stream)
                self.stream.flush()
            except Exception as e:
                # Fallback to basic print
                print(f"Alert: {message}")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _write)


class FileHandler(AlertHandler):
    """Handler for file-based alert logging."""
    
    def __init__(self, name: str = "file", config: Optional[Dict[str, Any]] = None):
        """
        Initialize file handler.
        
        Args:
            name: Handler name
            config: Handler configuration
        """
        config = config or {}
        super().__init__(name, config)
        
        # File-specific configuration
        self.file_path = Path(config.get('file_path', 'data/alerts.log'))
        self.max_file_size_mb = config.get('max_file_size_mb', 100)
        self.backup_count = config.get('backup_count', 5)
        self.file_format = config.get('file_format', 'json')  # json, text, csv
        self.include_detection_data = config.get('include_detection_data', True)
        self.auto_rotate = config.get('auto_rotate', True)
        
        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # File rotation tracking
        self._file_lock = threading.Lock()
        self._current_size = 0
        self._check_current_size()
    
    def _check_current_size(self):
        """Check current file size."""
        try:
            if self.file_path.exists():
                self._current_size = self.file_path.stat().st_size
            else:
                self._current_size = 0
        except Exception:
            self._current_size = 0
    
    async def _deliver_alert(self, alert: AlertMessage) -> bool:
        """Deliver alert to file."""
        try:
            # Format alert data
            formatted_data = self._format_alert_data(alert)
            
            # Write to file
            async with aiofiles.open(self.file_path, 'a', encoding='utf-8') as f:
                await f.write(formatted_data + '\n')
                await f.flush()
            
            # Update size tracking and check for rotation
            with self._file_lock:
                self._current_size += len(formatted_data.encode('utf-8')) + 1
                
                if self.auto_rotate and self._should_rotate():
                    await self._rotate_files()
            
            return True
        
        except Exception as e:
            self.logger.error(f"File handler error: {e}")
            return False
    
    def _format_alert_data(self, alert: AlertMessage) -> str:
        """Format alert data based on configured format."""
        if self.file_format == 'json':
            return self._format_as_json(alert)
        elif self.file_format == 'csv':
            return self._format_as_csv(alert)
        else:  # text format
            return self._format_as_text(alert)
    
    def _format_as_json(self, alert: AlertMessage) -> str:
        """Format alert as JSON."""
        data = {
            'timestamp': alert.created_at.isoformat(),
            'alert_id': alert.alert_id,
            'level': alert.alert_level.value,
            'message': alert.message,
            'source_id': alert.detection.source_id,
            'pattern_name': alert.detection.pattern_name,
            'confidence': alert.detection.confidence,
            'metadata': alert.metadata
        }
        
        if self.include_detection_data:
            data['detection'] = {
                'detection_id': str(alert.detection.detection_id),
                'frame_number': alert.detection.frame_number,
                'bounding_box': {
                    'x': alert.detection.bounding_box.x,
                    'y': alert.detection.bounding_box.y,
                    'width': alert.detection.bounding_box.width,
                    'height': alert.detection.bounding_box.height
                }
            }
        
        return json.dumps(data, separators=(',', ':'))
    
    def _format_as_csv(self, alert: AlertMessage) -> str:
        """Format alert as CSV."""
        fields = [
            alert.created_at.isoformat(),
            alert.alert_id,
            alert.alert_level.value,
            f'"{alert.message}"',  # Quote message to handle commas
            alert.detection.source_id,
            alert.detection.pattern_name,
            str(alert.detection.confidence),
            str(alert.detection.frame_number)
        ]
        
        if self.include_detection_data:
            bbox = alert.detection.bounding_box
            fields.extend([
                str(bbox.x), str(bbox.y), str(bbox.width), str(bbox.height)
            ])
        
        return ','.join(fields)
    
    def _format_as_text(self, alert: AlertMessage) -> str:
        """Format alert as human-readable text."""
        timestamp = alert.created_at.strftime('%Y-%m-%d %H:%M:%S')
        text = (
            f"{timestamp} [{alert.alert_level.value.upper()}] "
            f"{alert.detection.source_id}:{alert.detection.pattern_name} - "
            f"{alert.message} (confidence: {alert.detection.confidence:.2f})"
        )
        
        if alert.metadata:
            metadata_str = ", ".join([f"{k}={v}" for k, v in alert.metadata.items()])
            text += f" [{metadata_str}]"
        
        return text
    
    def _should_rotate(self) -> bool:
        """Check if file should be rotated."""
        max_size_bytes = self.max_file_size_mb * 1024 * 1024
        return self._current_size >= max_size_bytes
    
    async def _rotate_files(self):
        """Rotate log files."""
        try:
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = self.file_path.with_suffix(f'.{timestamp}.log')
            
            # Move current file to backup
            if self.file_path.exists():
                self.file_path.rename(backup_path)
            
            # Clean up old backups
            await self._cleanup_old_backups()
            
            # Reset size counter
            self._current_size = 0
            
            self.logger.info(f"Rotated alert log file to {backup_path}")
        
        except Exception as e:
            self.logger.error(f"Error rotating log files: {e}")
    
    async def _cleanup_old_backups(self):
        """Clean up old backup files."""
        try:
            # Find all backup files
            pattern = f"{self.file_path.stem}.*.log"
            backup_files = list(self.file_path.parent.glob(pattern))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            # Remove excess backups
            for backup_file in backup_files[self.backup_count:]:
                backup_file.unlink()
                self.logger.debug(f"Removed old backup: {backup_file}")
        
        except Exception as e:
            self.logger.error(f"Error cleaning up backups: {e}")


class WebhookHandler(AlertHandler):
    """Handler for HTTP webhook alert delivery."""
    
    def __init__(self, name: str = "webhook", config: Optional[Dict[str, Any]] = None):
        """
        Initialize webhook handler.
        
        Args:
            name: Handler name
            config: Handler configuration
        """
        config = config or {}
        super().__init__(name, config)
        
        # Webhook-specific configuration
        self.url = config.get('url', '')
        self.method = config.get('method', 'POST').upper()
        self.headers = config.get('headers', {'Content-Type': 'application/json'})
        self.auth = config.get('auth', None)  # Basic auth tuple (username, password)
        self.verify_ssl = config.get('verify_ssl', True)
        self.payload_template = config.get('payload_template', None)
        self.success_codes = config.get('success_codes', [200, 201, 202, 204])
        
        # Connection settings
        self.connect_timeout = config.get('connect_timeout', 10.0)
        self.read_timeout = config.get('read_timeout', 30.0)
        
        # Validate configuration
        if not self.url:
            raise ValueError("Webhook URL is required")
        
        # Parse URL to validate
        parsed = urlparse(self.url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid webhook URL: {self.url}")
    
    async def _deliver_alert(self, alert: AlertMessage) -> bool:
        """Deliver alert via HTTP webhook."""
        try:
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(
                connect=self.connect_timeout,
                total=self.timeout_seconds
            )
            
            connector = aiohttp.TCPConnector(verify_ssl=self.verify_ssl)
            
            async with aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers=self.headers
            ) as session:
                # Prepare payload
                payload = self._create_payload(alert)
                
                # Prepare authentication
                auth = None
                if self.auth:
                    auth = aiohttp.BasicAuth(self.auth[0], self.auth[1])
                
                # Make HTTP request
                async with session.request(
                    self.method,
                    self.url,
                    json=payload,
                    auth=auth
                ) as response:
                    # Check if response indicates success
                    if response.status in self.success_codes:
                        self.logger.debug(
                            f"Webhook delivered successfully: {response.status} {response.reason}"
                        )
                        return True
                    else:
                        response_text = await response.text()
                        self.logger.error(
                            f"Webhook delivery failed: {response.status} {response.reason} - {response_text}"
                        )
                        return False
        
        except aiohttp.ClientError as e:
            self.logger.error(f"Webhook client error: {e}")
            return False
        
        except asyncio.TimeoutError:
            self.logger.error(f"Webhook delivery timeout to {self.url}")
            return False
        
        except Exception as e:
            self.logger.error(f"Webhook handler error: {e}")
            return False
    
    def _create_payload(self, alert: AlertMessage) -> Dict[str, Any]:
        """Create webhook payload."""
        if self.payload_template:
            # Use custom template
            return {
                key: str(value).format(
                    alert_id=alert.alert_id,
                    level=alert.alert_level.value,
                    message=alert.message,
                    source_id=alert.detection.source_id,
                    pattern_name=alert.detection.pattern_name,
                    confidence=alert.detection.confidence,
                    timestamp=alert.created_at.isoformat(),
                    **alert.metadata
                ) for key, value in self.payload_template.items()
            }
        
        # Default payload structure
        return {
            'alert_id': alert.alert_id,
            'level': alert.alert_level.value,
            'message': alert.message,
            'timestamp': alert.created_at.isoformat(),
            'source': {
                'id': alert.detection.source_id,
                'name': alert.detection.source_name or alert.detection.source_id
            },
            'detection': {
                'pattern_name': alert.detection.pattern_name,
                'confidence': alert.detection.confidence,
                'frame_number': alert.detection.frame_number,
                'bounding_box': {
                    'x': alert.detection.bounding_box.x,
                    'y': alert.detection.bounding_box.y,
                    'width': alert.detection.bounding_box.width,
                    'height': alert.detection.bounding_box.height
                }
            },
            'metadata': alert.metadata
        }


class EmailHandler(AlertHandler):
    """Handler for email alert delivery."""
    
    def __init__(self, name: str = "email", config: Optional[Dict[str, Any]] = None):
        """
        Initialize email handler.
        
        Args:
            name: Handler name
            config: Handler configuration
        """
        config = config or {}
        super().__init__(name, config)
        
        # SMTP configuration
        self.smtp_server = config.get('smtp_server', 'localhost')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username', '')
        self.password = config.get('password', '')
        self.use_tls = config.get('use_tls', True)
        self.use_ssl = config.get('use_ssl', False)
        
        # Email configuration
        self.from_email = config.get('from_email', 'alerts@pyds-app.local')
        self.to_emails = config.get('to_emails', [])
        self.cc_emails = config.get('cc_emails', [])
        self.bcc_emails = config.get('bcc_emails', [])
        self.subject_template = config.get('subject_template', '[{level}] Alert from {source_id}: {pattern_name}')
        self.body_template = config.get('body_template', None)
        self.include_html = config.get('include_html', True)
        self.include_attachments = config.get('include_attachments', False)
        
        # Rate limiting for email
        self.min_interval_seconds = config.get('min_interval_seconds', 60.0)
        self._last_sent_time = 0.0
        
        # Validate configuration
        if not self.to_emails:
            raise ValueError("At least one recipient email is required")
        
        if not self.smtp_server:
            raise ValueError("SMTP server is required")
    
    async def _deliver_alert(self, alert: AlertMessage) -> bool:
        """Deliver alert via email."""
        try:
            # Check rate limiting
            current_time = time.time()
            if current_time - self._last_sent_time < self.min_interval_seconds:
                self.logger.debug("Email rate limited, skipping delivery")
                return False
            
            # Create email message
            message = await self._create_email_message(alert)
            
            # Send email
            success = await self._send_email(message)
            
            if success:
                self._last_sent_time = current_time
            
            return success
        
        except Exception as e:
            self.logger.error(f"Email handler error: {e}")
            return False
    
    async def _create_email_message(self, alert: AlertMessage) -> MIMEMultipart:
        """Create email message."""
        # Create message
        message = MIMEMultipart('alternative')
        
        # Set headers
        message['Subject'] = self._format_subject(alert)
        message['From'] = self.from_email
        message['To'] = ', '.join(self.to_emails)
        
        if self.cc_emails:
            message['Cc'] = ', '.join(self.cc_emails)
        
        # Create message body
        text_body = self._create_text_body(alert)
        text_part = MIMEText(text_body, 'plain', 'utf-8')
        message.attach(text_part)
        
        # Add HTML body if requested
        if self.include_html:
            html_body = self._create_html_body(alert)
            html_part = MIMEText(html_body, 'html', 'utf-8')
            message.attach(html_part)
        
        # Add attachments if requested
        if self.include_attachments:
            await self._add_attachments(message, alert)
        
        return message
    
    def _format_subject(self, alert: AlertMessage) -> str:
        """Format email subject."""
        return self.subject_template.format(
            level=alert.alert_level.value.upper(),
            source_id=alert.detection.source_id,
            pattern_name=alert.detection.pattern_name,
            confidence=alert.detection.confidence,
            timestamp=alert.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            **alert.metadata
        )
    
    def _create_text_body(self, alert: AlertMessage) -> str:
        """Create plain text email body."""
        if self.body_template:
            return self.body_template.format(
                alert_id=alert.alert_id,
                level=alert.alert_level.value,
                message=alert.message,
                source_id=alert.detection.source_id,
                pattern_name=alert.detection.pattern_name,
                confidence=alert.detection.confidence,
                timestamp=alert.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                frame_number=alert.detection.frame_number,
                **alert.metadata
            )
        
        # Default text body
        return f"""
Alert Details:
==============

Alert ID: {alert.alert_id}
Level: {alert.alert_level.value.upper()}
Message: {alert.message}
Timestamp: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}

Detection Information:
---------------------
Source: {alert.detection.source_id}
Pattern: {alert.detection.pattern_name}
Confidence: {alert.detection.confidence:.2f}
Frame: {alert.detection.frame_number}

Bounding Box:
X: {alert.detection.bounding_box.x:.3f}
Y: {alert.detection.bounding_box.y:.3f}
Width: {alert.detection.bounding_box.width:.3f}
Height: {alert.detection.bounding_box.height:.3f}

Additional Metadata:
{json.dumps(alert.metadata, indent=2) if alert.metadata else 'None'}

---
Generated by PyDS Alert System
""".strip()
    
    def _create_html_body(self, alert: AlertMessage) -> str:
        """Create HTML email body."""
        # Color mapping for alert levels
        level_colors = {
            AlertLevel.LOW: '#17a2b8',      # Info blue
            AlertLevel.MEDIUM: '#ffc107',   # Warning yellow
            AlertLevel.HIGH: '#fd7e14',     # Orange
            AlertLevel.CRITICAL: '#dc3545', # Danger red
        }
        
        level_color = level_colors.get(alert.alert_level, '#6c757d')
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: {level_color}; color: white; padding: 15px; border-radius: 5px; }}
        .content {{ margin: 20px 0; }}
        .info-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        .info-table th, .info-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .info-table th {{ background-color: #f2f2f2; }}
        .footer {{ color: #666; font-size: 12px; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>[{alert.alert_level.value.upper()}] Alert Notification</h2>
        <p>{alert.message}</p>
    </div>
    
    <div class="content">
        <h3>Alert Details</h3>
        <table class="info-table">
            <tr><th>Alert ID</th><td>{alert.alert_id}</td></tr>
            <tr><th>Timestamp</th><td>{alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
            <tr><th>Source</th><td>{alert.detection.source_id}</td></tr>
            <tr><th>Pattern</th><td>{alert.detection.pattern_name}</td></tr>
            <tr><th>Confidence</th><td>{alert.detection.confidence:.2%}</td></tr>
            <tr><th>Frame Number</th><td>{alert.detection.frame_number}</td></tr>
        </table>
        
        <h3>Detection Location</h3>
        <table class="info-table">
            <tr><th>X</th><td>{alert.detection.bounding_box.x:.3f}</td></tr>
            <tr><th>Y</th><td>{alert.detection.bounding_box.y:.3f}</td></tr>
            <tr><th>Width</th><td>{alert.detection.bounding_box.width:.3f}</td></tr>
            <tr><th>Height</th><td>{alert.detection.bounding_box.height:.3f}</td></tr>
        </table>
        
        {self._format_metadata_html(alert.metadata)}
    </div>
    
    <div class="footer">
        <p>Generated by PyDS Alert System</p>
    </div>
</body>
</html>
""".strip()
    
    def _format_metadata_html(self, metadata: Dict[str, Any]) -> str:
        """Format metadata as HTML table."""
        if not metadata:
            return ""
        
        rows = []
        for key, value in metadata.items():
            rows.append(f"<tr><th>{key}</th><td>{value}</td></tr>")
        
        return f"""
        <h3>Additional Metadata</h3>
        <table class="info-table">
            {"".join(rows)}
        </table>
        """
    
    async def _add_attachments(self, message: MIMEMultipart, alert: AlertMessage):
        """Add attachments to email message."""
        try:
            # Create a JSON attachment with full alert data
            alert_data = alert.to_dict()
            json_data = json.dumps(alert_data, indent=2)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(json_data)
                temp_path = f.name
            
            # Read file and attach
            with open(temp_path, 'rb') as f:
                attachment = MIMEBase('application', 'json')
                attachment.set_payload(f.read())
                encoders.encode_base64(attachment)
                
                attachment.add_header(
                    'Content-Disposition',
                    f'attachment; filename="alert_{alert.alert_id}.json"'
                )
                message.attach(attachment)
            
            # Clean up temporary file
            os.unlink(temp_path)
        
        except Exception as e:
            self.logger.error(f"Error adding email attachments: {e}")
    
    async def _send_email(self, message: MIMEMultipart) -> bool:
        """Send email message."""
        def _send():
            try:
                # Create SMTP connection
                if self.use_ssl:
                    server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
                else:
                    server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                
                # Enable TLS if requested
                if self.use_tls and not self.use_ssl:
                    server.starttls(context=ssl.create_default_context())
                
                # Login if credentials provided
                if self.username and self.password:
                    server.login(self.username, self.password)
                
                # Send message
                recipients = self.to_emails + self.cc_emails + self.bcc_emails
                server.send_message(message, to_addrs=recipients)
                server.quit()
                
                return True
            
            except Exception as e:
                self.logger.error(f"SMTP error: {e}")
                return False
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _send)


class SlackHandler(AlertHandler):
    """Handler for Slack webhook notifications."""
    
    def __init__(self, name: str = "slack", config: Optional[Dict[str, Any]] = None):
        """
        Initialize Slack handler.
        
        Args:
            name: Handler name
            config: Handler configuration
        """
        config = config or {}
        super().__init__(name, config)
        
        # Slack-specific configuration
        self.webhook_url = config.get('webhook_url', '')
        self.channel = config.get('channel', None)
        self.username = config.get('username', 'PyDS Alert System')
        self.icon_emoji = config.get('icon_emoji', ':warning:')
        self.mention_users = config.get('mention_users', [])
        self.mention_channels = config.get('mention_channels', [])
        
        # Alert level to color mapping
        self.level_colors = {
            AlertLevel.LOW: '#36a64f',      # Green
            AlertLevel.MEDIUM: '#ffb347',   # Orange
            AlertLevel.HIGH: '#ff6b6b',     # Red
            AlertLevel.CRITICAL: '#d63031', # Dark red
        }
        
        if not self.webhook_url:
            raise ValueError("Slack webhook URL is required")
    
    async def _deliver_alert(self, alert: AlertMessage) -> bool:
        """Deliver alert to Slack."""
        try:
            # Create Slack payload
            payload = self._create_slack_payload(alert)
            
            # Send to Slack
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    return response.status == 200
        
        except Exception as e:
            self.logger.error(f"Slack handler error: {e}")
            return False
    
    def _create_slack_payload(self, alert: AlertMessage) -> Dict[str, Any]:
        """Create Slack message payload."""
        # Create mention string
        mentions = []
        for user in self.mention_users:
            mentions.append(f"<@{user}>")
        for channel in self.mention_channels:
            mentions.append(f"<!{channel}>")
        
        mention_str = " ".join(mentions)
        if mention_str:
            mention_str += " "
        
        # Create main message
        text = f"{mention_str}*Alert: {alert.message}*"
        
        # Create attachment with details
        attachment = {
            "color": self.level_colors.get(alert.alert_level, "#cccccc"),
            "title": f"{alert.alert_level.value.upper()} Alert",
            "fields": [
                {
                    "title": "Source",
                    "value": alert.detection.source_id,
                    "short": True
                },
                {
                    "title": "Pattern",
                    "value": alert.detection.pattern_name,
                    "short": True
                },
                {
                    "title": "Confidence",
                    "value": f"{alert.detection.confidence:.2%}",
                    "short": True
                },
                {
                    "title": "Frame",
                    "value": str(alert.detection.frame_number),
                    "short": True
                },
                {
                    "title": "Timestamp",
                    "value": alert.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                    "short": False
                }
            ],
            "footer": "PyDS Alert System",
            "ts": int(alert.created_at.timestamp())
        }
        
        # Add metadata fields
        if alert.metadata:
            for key, value in alert.metadata.items():
                attachment["fields"].append({
                    "title": key.replace('_', ' ').title(),
                    "value": str(value),
                    "short": True
                })
        
        payload = {
            "text": text,
            "attachments": [attachment],
            "username": self.username,
            "icon_emoji": self.icon_emoji
        }
        
        if self.channel:
            payload["channel"] = self.channel
        
        return payload


# Handler registry for dynamic handler creation
HANDLER_REGISTRY = {
    'console': ConsoleHandler,
    'file': FileHandler,
    'webhook': WebhookHandler,
    'email': EmailHandler,
    'slack': SlackHandler,
}


def create_handler(handler_type: str, name: str, config: Dict[str, Any]) -> AlertHandler:
    """
    Create alert handler by type.
    
    Args:
        handler_type: Type of handler to create
        name: Handler name
        config: Handler configuration
        
    Returns:
        Created handler instance
        
    Raises:
        ValueError: If handler type is not supported
    """
    if handler_type not in HANDLER_REGISTRY:
        available_types = ', '.join(HANDLER_REGISTRY.keys())
        raise ValueError(f"Unsupported handler type: {handler_type}. Available: {available_types}")
    
    handler_class = HANDLER_REGISTRY[handler_type]
    return handler_class(name, config)


def register_custom_handler(handler_type: str, handler_class: type):
    """
    Register custom handler class.
    
    Args:
        handler_type: Type name for the handler
        handler_class: Handler class (must inherit from AlertHandler)
    """
    if not issubclass(handler_class, AlertHandler):
        raise ValueError("Handler class must inherit from AlertHandler")
    
    HANDLER_REGISTRY[handler_type] = handler_class


def get_available_handler_types() -> List[str]:
    """Get list of available handler types."""
    return list(HANDLER_REGISTRY.keys())


async def create_default_handlers(config: AppConfig) -> List[AlertHandler]:
    """
    Create default alert handlers based on configuration.
    
    Args:
        config: Application configuration
        
    Returns:
        List of created handlers
    """
    handlers = []
    
    try:
        # Always create console handler
        console_handler = ConsoleHandler("console", {
            'colored_output': True,
            'include_timestamp': True,
            'include_metadata': True
        })
        handlers.append(console_handler)
        
        # Create file handler if configured
        if hasattr(config.alerts, 'file_output') and config.alerts.file_output:
            file_handler = FileHandler("file", {
                'file_path': 'data/alerts.log',
                'file_format': 'json',
                'max_file_size_mb': 50,
                'backup_count': 3
            })
            handlers.append(file_handler)
        
        # Create webhook handler if configured
        webhook_config = getattr(config.alerts, 'webhook', None)
        if webhook_config and webhook_config.get('enabled'):
            webhook_handler = WebhookHandler("webhook", webhook_config)
            handlers.append(webhook_handler)
        
        # Create email handler if configured
        email_config = getattr(config.alerts, 'email', None)
        if email_config and email_config.get('enabled'):
            email_handler = EmailHandler("email", email_config)
            handlers.append(email_handler)
        
        return handlers
    
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error creating default handlers: {e}")
        # Return at least console handler
        return [ConsoleHandler("console")]
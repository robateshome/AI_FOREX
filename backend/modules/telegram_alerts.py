#!/usr/bin/env python3
"""
Telegram Alerts Module
Sends trading signals and performance updates via Telegram bot
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class TelegramConfig:
    """Configuration for Telegram bot"""
    bot_token: str
    chat_id: str
    enable_alerts: bool = True
    alert_types: List[str] = None
    
    def __post_init__(self):
        if self.alert_types is None:
            self.alert_types = ['signals', 'performance', 'errors']

@dataclass
class AlertMessage:
    """Structure for alert messages"""
    type: str  # 'signal', 'performance', 'error', 'info'
    title: str
    message: str
    timestamp: datetime
    priority: str = 'normal'  # 'low', 'normal', 'high', 'urgent'
    data: Optional[Dict[str, Any]] = None

class TelegramAlerts:
    """Telegram bot for sending trading alerts"""
    
    def __init__(self, config: TelegramConfig):
        self.config = config
        self.base_url = f"https://api.telegram.org/bot{config.bot_token}"
        self.session = None
        self.is_running = False
        self.alert_queue = asyncio.Queue()
        self.rate_limit_delay = 0.1  # 100ms between messages
        
        # Alert templates
        self.templates = {
            'signal': self._format_signal_alert,
            'performance': self._format_performance_alert,
            'error': self._format_error_alert,
            'info': self._format_info_alert
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def start(self):
        """Start the Telegram alerts service"""
        if not self.config.enable_alerts:
            logger.info("Telegram alerts disabled")
            return
        
        self.is_running = True
        logger.info("Telegram alerts service started")
        
        # Start alert processor
        asyncio.create_task(self._process_alert_queue())
        
        # Send startup message
        await self.send_info_alert("🚀 AI Forex Signal Generator Started", 
                                 "The system is now running and monitoring markets.")
    
    async def stop(self):
        """Stop the Telegram alerts service"""
        self.is_running = False
        logger.info("Telegram alerts service stopped")
        
        # Send shutdown message
        await self.send_info_alert("🛑 AI Forex Signal Generator Stopped", 
                                 "The system has been shut down.")
    
    async def send_signal_alert(self, signal_data: Dict[str, Any]):
        """Send trading signal alert"""
        if not self.config.enable_alerts or 'signals' not in self.config.alert_types:
            return
        
        try:
            alert = AlertMessage(
                type='signal',
                title=f"🎯 Trading Signal: {signal_data.get('action', 'UNKNOWN').upper()}",
                message=self._format_signal_message(signal_data),
                timestamp=datetime.utcnow(),
                priority='high',
                data=signal_data
            )
            
            await self.alert_queue.put(alert)
            logger.debug(f"Signal alert queued for {signal_data.get('symbol', 'UNKNOWN')}")
            
        except Exception as e:
            logger.error(f"Error queuing signal alert: {e}")
    
    async def send_performance_alert(self, performance_data: Dict[str, Any]):
        """Send performance update alert"""
        if not self.config.enable_alerts or 'performance' not in self.config.alert_types:
            return
        
        try:
            alert = AlertMessage(
                type='performance',
                title="📊 Performance Update",
                message=self._format_performance_message(performance_data),
                timestamp=datetime.utcnow(),
                priority='normal',
                data=performance_data
            )
            
            await self.alert_queue.put(alert)
            logger.debug("Performance alert queued")
            
        except Exception as e:
            logger.error(f"Error queuing performance alert: {e}")
    
    async def send_error_alert(self, error_msg: str, error_details: Optional[Dict[str, Any]] = None):
        """Send error alert"""
        if not self.config.enable_alerts or 'errors' not in self.config.alert_types:
            return
        
        try:
            alert = AlertMessage(
                type='error',
                title="❌ System Error",
                message=error_msg,
                timestamp=datetime.utcnow(),
                priority='urgent',
                data=error_details
            )
            
            await self.alert_queue.put(alert)
            logger.debug("Error alert queued")
            
        except Exception as e:
            logger.error(f"Error queuing error alert: {e}")
    
    async def send_info_alert(self, title: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Send informational alert"""
        if not self.config.enable_alerts:
            return
        
        try:
            alert = AlertMessage(
                type='info',
                title=title,
                message=message,
                timestamp=datetime.utcnow(),
                priority='low',
                data=data
            )
            
            await self.alert_queue.put(alert)
            logger.debug("Info alert queued")
            
        except Exception as e:
            logger.error(f"Error queuing info alert: {e}")
    
    async def _process_alert_queue(self):
        """Process alerts from the queue"""
        while self.is_running:
            try:
                # Get alert from queue
                alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)
                
                # Format and send the alert
                await self._send_alert(alert)
                
                # Rate limiting
                await asyncio.sleep(self.rate_limit_delay)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing alert queue: {e}")
    
    async def _send_alert(self, alert: AlertMessage):
        """Send a single alert via Telegram"""
        try:
            # Format message based on type
            if alert.type in self.templates:
                formatted_message = self.templates[alert.type](alert)
            else:
                formatted_message = self._format_generic_alert(alert)
            
            # Send via Telegram API
            await self._send_telegram_message(formatted_message)
            
            logger.info(f"Alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
    
    async def _send_telegram_message(self, message: str):
        """Send message via Telegram Bot API"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.config.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            async with self.session.post(url, json=data) as response:
                if response.status != 200:
                    response_text = await response.text()
                    logger.error(f"Telegram API error {response.status}: {response_text}")
                else:
                    logger.debug("Message sent successfully")
                    
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
    
    def _format_signal_alert(self, alert: AlertMessage) -> str:
        """Format trading signal alert"""
        signal_data = alert.data
        if not signal_data:
            return "Invalid signal data"
        
        symbol = signal_data.get('symbol', 'UNKNOWN')
        action = signal_data.get('action', 'UNKNOWN').upper()
        confidence = signal_data.get('confidence', 0)
        price = signal_data.get('price', 0)
        timestamp = signal_data.get('timestamp', datetime.utcnow())
        
        # Format timestamp
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Confidence emoji
        confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
        
        # Action emoji
        action_emoji = "📈" if action == "BUY" else "📉" if action == "SELL" else "⏸️"
        
        message = f"""
{action_emoji} <b>{action} SIGNAL</b> {confidence_emoji}

<b>Symbol:</b> {symbol}
<b>Price:</b> {price:.5f}
<b>Confidence:</b> {confidence:.2%}
<b>Time:</b> {time_str}

<b>Model:</b> {signal_data.get('model_version', 'Unknown')}
        """.strip()
        
        return message
    
    def _format_performance_alert(self, alert: AlertMessage) -> str:
        """Format performance alert"""
        perf_data = alert.data
        if not perf_data:
            return "Invalid performance data"
        
        message = f"""
📊 <b>Performance Update</b>

<b>Accuracy:</b> {perf_data.get('accuracy', 0):.2%}
<b>Sharpe Ratio:</b> {perf_data.get('sharpe_ratio', 0):.3f}
<b>Win Rate:</b> {perf_data.get('win_rate', 0):.2%}
<b>Max Drawdown:</b> {perf_data.get('max_drawdown', 0):.2%}
<b>Total Trades:</b> {perf_data.get('total_trades', 0)}

<b>Last Signal:</b> {perf_data.get('last_signal_action', 'None')}
<b>Live Signals:</b> {perf_data.get('live_signals_count', 0)}
        """.strip()
        
        return message
    
    def _format_error_alert(self, alert: AlertMessage) -> str:
        """Format error alert"""
        message = f"""
❌ <b>System Error</b>

<b>Error:</b> {alert.message}
<b>Time:</b> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

<b>Priority:</b> {alert.priority.upper()}
        """.strip()
        
        if alert.data:
            message += f"\n\n<b>Details:</b>\n{json.dumps(alert.data, indent=2)}"
        
        return message
    
    def _format_info_alert(self, alert: AlertMessage) -> str:
        """Format info alert"""
        message = f"""
ℹ️ <b>{alert.title}</b>

{alert.message}

<b>Time:</b> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
        """.strip()
        
        return message
    
    def _format_generic_alert(self, alert: AlertMessage) -> str:
        """Format generic alert"""
        message = f"""
📢 <b>{alert.title}</b>

{alert.message}

<b>Type:</b> {alert.type}
<b>Priority:</b> {alert.priority}
<b>Time:</b> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
        """.strip()
        
        return message
    
    def _format_signal_message(self, signal_data: Dict[str, Any]) -> str:
        """Format signal message for alert"""
        return f"Signal generated for {signal_data.get('symbol', 'UNKNOWN')}: {signal_data.get('action', 'UNKNOWN')} at {signal_data.get('price', 0):.5f}"
    
    def _format_performance_message(self, perf_data: Dict[str, Any]) -> str:
        """Format performance message for alert"""
        return f"Performance update: {perf_data.get('accuracy', 0):.2%} accuracy, {perf_data.get('sharpe_ratio', 0):.3f} Sharpe ratio"
    
    async def test_connection(self) -> bool:
        """Test Telegram bot connection"""
        try:
            url = f"{self.base_url}/getMe"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('ok'):
                        bot_info = data['result']
                        logger.info(f"Telegram bot connected: @{bot_info.get('username', 'Unknown')}")
                        return True
                    else:
                        logger.error(f"Telegram API error: {data.get('description', 'Unknown error')}")
                        return False
                else:
                    logger.error(f"Telegram API HTTP error: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error testing Telegram connection: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            'is_running': self.is_running,
            'enable_alerts': self.config.enable_alerts,
            'alert_types': self.config.alert_types,
            'queue_size': self.alert_queue.qsize(),
            'rate_limit_delay': self.rate_limit_delay
        }

# Factory function to create alerts from environment
def create_telegram_alerts() -> TelegramAlerts:
    """Create Telegram alerts from environment variables"""
    load_dotenv()
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token:
        logger.warning("TELEGRAM_BOT_TOKEN not set, alerts will be disabled")
        return None
    
    if not chat_id:
        logger.warning("TELEGRAM_CHAT_ID not set, alerts will be disabled")
        return None
    
    config = TelegramConfig(
        bot_token=bot_token,
        chat_id=chat_id,
        enable_alerts=True
    )
    
    return TelegramAlerts(config)
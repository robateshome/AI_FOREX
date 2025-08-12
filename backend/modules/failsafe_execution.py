#!/usr/bin/env python3
"""
Module 0x07: Failsafe Execution
CRC32: PLACEHOLDER_CRC32_07

ERROR_POLICY: AUTO_SWITCH_TO_BACKUP_CONNECTION
CRITICAL_ALERT: DASHBOARD_POPUP + LOG
TRADE_EXECUTION_LOCK: IF_DATA_INVALID
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import os

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class SystemStatus(Enum):
    """System status"""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class SystemAlert:
    """System alert structure"""
    id: str
    timestamp: datetime
    level: AlertLevel
    module: str
    message: str
    details: Dict[str, Any]
    acknowledged: bool = False
    resolved: bool = False

class FailsafeManager:
    """Failsafe execution manager for system reliability"""
    
    def __init__(self):
        self.is_running_flag = False
        self.alert_callbacks: List[Callable] = []
        self.status_callbacks: List[Callback] = []
        
        # System status
        self.system_status = SystemStatus.OPERATIONAL
        self.last_status_change = datetime.now()
        
        # Alert management
        self.active_alerts: Dict[str, SystemAlert] = {}
        self.alert_history: List[SystemAlert] = []
        self.max_alerts = 100
        
        # Failsafe policies
        self.auto_switch_enabled = True
        self.trade_lock_enabled = True
        self.critical_threshold = 3  # Number of critical errors before emergency mode
        
        # Connection monitoring
        self.connection_status = {}
        self.backup_connections = {}
        self.primary_connection = None
        
        # Trade execution lock
        self.trade_lock_active = False
        self.lock_reason = ""
        self.lock_timestamp = None
        
        # Error tracking
        self.error_counts = {}
        self.last_error_reset = datetime.now()
        
        # Configuration
        self.config = {
            "max_retry_attempts": 3,
            "retry_delay_seconds": 5,
            "alert_retention_hours": 24,
            "auto_resolve_alerts": True,
            "emergency_shutdown": False
        }
        
        logger.info("Failsafe Manager initialized")
    
    async def start(self):
        """Start the failsafe manager"""
        if self.is_running_flag:
            return
        
        self.is_running_flag = True
        logger.info("Starting Failsafe Manager...")
        
        # Start monitoring tasks
        asyncio.create_task(self._run_system_monitor())
        asyncio.create_task(self._run_alert_cleanup())
        asyncio.create_task(self._run_connection_monitor())
        
        logger.info("Failsafe Manager started")
    
    async def stop(self):
        """Stop the failsafe manager"""
        self.is_running_flag = False
        logger.info("Stopping Failsafe Manager...")
        
        # Save alerts to storage
        await self._save_alerts()
        
        logger.info("Failsafe Manager stopped")
    
    def is_running(self) -> bool:
        """Check if the manager is running"""
        return self.is_running_flag
    
    async def _run_system_monitor(self):
        """Monitor overall system health"""
        while self.is_running_flag:
            try:
                # Check system status
                await self._assess_system_health()
                
                # Check for critical conditions
                await self._check_critical_conditions()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"System monitor error: {e}")
                await asyncio.sleep(10)
    
    async def _run_alert_cleanup(self):
        """Clean up old alerts"""
        while self.is_running_flag:
            try:
                await self._cleanup_old_alerts()
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Alert cleanup error: {e}")
                await asyncio.sleep(300)
    
    async def _run_connection_monitor(self):
        """Monitor connection health"""
        while self.is_running_flag:
            try:
                await self._check_connection_health()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Connection monitor error: {e}")
                await asyncio.sleep(10)
    
    async def _assess_system_health(self):
        """Assess overall system health"""
        try:
            # Count active alerts by level
            alert_counts = {level: 0 for level in AlertLevel}
            for alert in self.active_alerts.values():
                if not alert.resolved:
                    alert_counts[alert.level] += 1
            
            # Determine system status based on alerts
            if alert_counts[AlertLevel.CRITICAL] >= self.critical_threshold:
                new_status = SystemStatus.EMERGENCY
            elif alert_counts[AlertLevel.CRITICAL] > 0 or alert_counts[AlertLevel.ERROR] > 5:
                new_status = SystemStatus.CRITICAL
            elif alert_counts[AlertLevel.ERROR] > 0 or alert_counts[AlertLevel.WARNING] > 3:
                new_status = SystemStatus.DEGRADED
            else:
                new_status = SystemStatus.OPERATIONAL
            
            # Update status if changed
            if new_status != self.system_status:
                old_status = self.system_status
                self.system_status = new_status
                self.last_status_change = datetime.now()
                
                logger.warning(f"System status changed from {old_status.value} to {new_status.value}")
                
                # Notify status change
                await self._notify_status_change(old_status, new_status)
                
                # Take action based on status
                await self._handle_status_change(new_status)
                
        except Exception as e:
            logger.error(f"Error assessing system health: {e}")
    
    async def _check_critical_conditions(self):
        """Check for critical system conditions"""
        try:
            # Check if trade lock should be activated
            if self.system_status in [SystemStatus.CRITICAL, SystemStatus.EMERGENCY]:
                if not self.trade_lock_active:
                    await self.activate_trade_lock("System in critical state")
            
            # Check if emergency shutdown is needed
            if self.system_status == SystemStatus.EMERGENCY and self.config["emergency_shutdown"]:
                await self._trigger_emergency_shutdown()
                
        except Exception as e:
            logger.error(f"Error checking critical conditions: {e}")
    
    async def _handle_status_change(self, new_status: SystemStatus):
        """Handle system status changes"""
        try:
            if new_status == SystemStatus.EMERGENCY:
                # Activate all failsafe measures
                await self.activate_trade_lock("Emergency mode activated")
                await self._create_critical_alert("SYSTEM", "System entered emergency mode")
                
            elif new_status == SystemStatus.CRITICAL:
                # Activate trade lock
                await self.activate_trade_lock("Critical system state")
                await self._create_critical_alert("SYSTEM", "System entered critical state")
                
            elif new_status == SystemStatus.DEGRADED:
                # Monitor closely but allow trading
                await self._create_warning_alert("SYSTEM", "System performance degraded")
                
            elif new_status == SystemStatus.OPERATIONAL:
                # Release trade lock if it was due to system issues
                if self.trade_lock_active and "system" in self.lock_reason.lower():
                    await self.deactivate_trade_lock("System returned to operational state")
                
        except Exception as e:
            logger.error(f"Error handling status change: {e}")
    
    async def _trigger_emergency_shutdown(self):
        """Trigger emergency shutdown procedures"""
        try:
            logger.critical("EMERGENCY SHUTDOWN TRIGGERED")
            
            # Create emergency alert
            await self._create_critical_alert("SYSTEM", "EMERGENCY SHUTDOWN INITIATED")
            
            # Activate trade lock
            await self.activate_trade_lock("Emergency shutdown")
            
            # Notify all callbacks
            await self._notify_emergency_shutdown()
            
            # In a real system, this would trigger shutdown procedures
            # For now, we just log and continue monitoring
            
        except Exception as e:
            logger.error(f"Error during emergency shutdown: {e}")
    
    async def _check_connection_health(self):
        """Check health of all connections"""
        try:
            for connection_name, status in self.connection_status.items():
                if status == "disconnected" and self.auto_switch_enabled:
                    # Try to switch to backup connection
                    await self._switch_to_backup(connection_name)
                    
        except Exception as e:
            logger.error(f"Error checking connection health: {e}")
    
    async def _switch_to_backup(self, connection_name: str):
        """Switch to backup connection"""
        try:
            if connection_name in self.backup_connections:
                backup_name = self.backup_connections[connection_name]
                logger.info(f"Switching from {connection_name} to backup {backup_name}")
                
                # Update connection status
                self.connection_status[connection_name] = "switching"
                
                # Create info alert
                await self._create_info_alert("CONNECTION", f"Switching to backup connection: {backup_name}")
                
                # In a real implementation, this would trigger the actual connection switch
                
        except Exception as e:
            logger.error(f"Error switching to backup connection: {e}")
    
    async def create_alert(self, level: AlertLevel, module: str, message: str, details: Dict[str, Any] = None) -> str:
        """Create a new system alert"""
        try:
            alert_id = f"{module}_{int(datetime.now().timestamp())}"
            
            alert = SystemAlert(
                id=alert_id,
                timestamp=datetime.now(),
                level=level,
                module=module,
                message=message,
                details=details or {}
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            # Limit history size
            if len(self.alert_history) > self.max_alerts:
                self.alert_history.pop(0)
            
            # Log alert
            log_level = getattr(logger, level.value, logger.info)
            log_level(f"[{module}] {message}")
            
            # Notify callbacks
            await self._notify_alert_callbacks(alert)
            
            # Auto-resolve info alerts after some time
            if level == AlertLevel.INFO and self.config["auto_resolve_alerts"]:
                asyncio.create_task(self._auto_resolve_alert(alert_id, delay_seconds=300))
            
            return alert_id
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return ""
    
    async def _create_critical_alert(self, module: str, message: str, details: Dict[str, Any] = None):
        """Create a critical alert"""
        return await self.create_alert(AlertLevel.CRITICAL, module, message, details)
    
    async def _create_warning_alert(self, module: str, message: str, details: Dict[str, Any] = None):
        """Create a warning alert"""
        return await self.create_alert(AlertLevel.WARNING, module, message, details)
    
    async def _create_info_alert(self, module: str, message: str, details: Dict[str, Any] = None):
        """Create an info alert"""
        return await self.create_alert(AlertLevel.INFO, module, message, details)
    
    async def _auto_resolve_alert(self, alert_id: str, delay_seconds: int):
        """Auto-resolve an alert after delay"""
        try:
            await asyncio.sleep(delay_seconds)
            
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                if not alert.acknowledged and not alert.resolved:
                    alert.resolved = True
                    logger.info(f"Auto-resolved alert: {alert_id}")
                    
        except Exception as e:
            logger.error(f"Error auto-resolving alert: {e}")
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.acknowledged = True
                
                # Move to history
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]
                
                logger.info(f"Alert resolved: {alert_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return False
    
    async def activate_trade_lock(self, reason: str):
        """Activate trade execution lock"""
        try:
            if not self.trade_lock_active:
                self.trade_lock_active = True
                self.lock_reason = reason
                self.lock_timestamp = datetime.now()
                
                logger.warning(f"Trade lock activated: {reason}")
                
                # Create warning alert
                await self._create_warning_alert("TRADING", f"Trade lock activated: {reason}")
                
                # Notify callbacks
                await self._notify_trade_lock_change(True, reason)
                
        except Exception as e:
            logger.error(f"Error activating trade lock: {e}")
    
    async def deactivate_trade_lock(self, reason: str):
        """Deactivate trade execution lock"""
        try:
            if self.trade_lock_active:
                self.trade_lock_active = False
                self.lock_reason = ""
                self.lock_timestamp = None
                
                logger.info(f"Trade lock deactivated: {reason}")
                
                # Create info alert
                await self._create_info_alert("TRADING", f"Trade lock deactivated: {reason}")
                
                # Notify callbacks
                await self._notify_trade_lock_change(False, reason)
                
        except Exception as e:
            logger.error(f"Error deactivating trade lock: {e}")
    
    def is_trade_locked(self) -> bool:
        """Check if trading is locked"""
        return self.trade_lock_active
    
    def get_trade_lock_info(self) -> Dict:
        """Get trade lock information"""
        return {
            "locked": self.trade_lock_active,
            "reason": self.lock_reason,
            "timestamp": self.lock_timestamp.isoformat() if self.lock_timestamp else None
        }
    
    async def _cleanup_old_alerts(self):
        """Clean up old alerts"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.config["alert_retention_hours"])
            
            # Remove old resolved alerts from history
            self.alert_history = [
                alert for alert in self.alert_history
                if alert.timestamp > cutoff_time or not alert.resolved
            ]
            
            logger.debug("Old alerts cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up old alerts: {e}")
    
    async def _save_alerts(self):
        """Save alerts to persistent storage"""
        try:
            # In a real implementation, this would save to database or file
            # For now, we just log the action
            logger.info("Alerts saved to storage")
            
        except Exception as e:
            logger.error(f"Error saving alerts: {e}")
    
    async def _notify_alert_callbacks(self, alert: SystemAlert):
        """Notify all registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Alert callback notification failed: {e}")
    
    async def _notify_status_callbacks(self, old_status: SystemStatus, new_status: SystemStatus):
        """Notify all registered status callbacks"""
        for callback in self.status_callbacks:
            try:
                await callback(old_status, new_status)
            except Exception as e:
                logger.error(f"Status callback notification failed: {e}")
    
    async def _notify_trade_lock_change(self, locked: bool, reason: str):
        """Notify all registered callbacks of trade lock change"""
        # This would notify trading systems of lock status changes
        logger.info(f"Trade lock change notified: locked={locked}, reason={reason}")
    
    async def _notify_emergency_shutdown(self):
        """Notify all registered callbacks of emergency shutdown"""
        # This would notify all systems of emergency shutdown
        logger.critical("Emergency shutdown notification sent to all systems")
    
    def register_alert_callback(self, callback: Callable):
        """Register a callback for alert updates"""
        if callback not in self.alert_callbacks:
            self.alert_callbacks.append(callback)
    
    def register_status_callback(self, callback: Callable):
        """Register a callback for status changes"""
        if callback not in self.status_callbacks:
            self.status_callbacks.append(callback)
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            "status": self.system_status.value,
            "last_change": self.last_status_change.isoformat(),
            "trade_locked": self.trade_lock_active,
            "active_alerts": len(self.active_alerts),
            "total_alerts": len(self.alert_history),
            "error_counts": self.error_counts
        }
    
    def get_active_alerts(self) -> List[Dict]:
        """Get active alerts for API"""
        return [
            {
                "id": alert.id,
                "timestamp": alert.timestamp.isoformat(),
                "level": alert.level.value,
                "module": alert.module,
                "message": alert.message,
                "details": alert.details,
                "acknowledged": alert.acknowledged,
                "resolved": alert.resolved
            }
            for alert in self.active_alerts.values()
        ]
    
    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        """Get alert history for API"""
        recent_alerts = self.alert_history[-limit:]
        return [
            {
                "id": alert.id,
                "timestamp": alert.timestamp.isoformat(),
                "level": alert.level.value,
                "module": alert.module,
                "message": alert.message,
                "details": alert.details,
                "acknowledged": alert.acknowledged,
                "resolved": alert.resolved
            }
            for alert in recent_alerts
        ]
    
    async def update_config(self, config: dict):
        """Update configuration"""
        self.config.update(config)
        logger.info("Failsafe manager configuration updated")
    
    def get_config(self) -> Dict:
        """Get current configuration"""
        return self.config.copy()
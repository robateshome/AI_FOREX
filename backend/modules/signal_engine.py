#!/usr/bin/env python3
"""
Module 0x03: Signal Engine
CRC32: PLACEHOLDER_CRC32_03

FORMAT: BINARY_SIGNAL + TIMESTAMP + SYMBOL + DIVERGENCE_TYPE + CONFIDENCE_LEVEL + PRICE
TRANSPORT: WEBSOCKET_SERVER_PUSH + REST_API_PULL + LOCAL_BROADCAST
LOG: SQLITE_TRADE_HISTORY
EXECUTION: HOOK_READY_FOR_ORDER_ROUTING (LIVE_MODE_DEFAULT)
"""

import asyncio
import logging
import sqlite3
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import os

logger = logging.getLogger(__name__)

class SignalStatus(Enum):
    """Signal status enumeration"""
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

@dataclass
class TradingSignal:
    """Trading signal structure"""
    id: str
    timestamp: datetime
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    divergence_type: str
    confidence_level: float
    price: float
    status: SignalStatus
    indicators: Dict[str, float]
    description: str
    execution_price: Optional[float] = None
    execution_time: Optional[datetime] = None
    pnl: Optional[float] = None

class SignalEngine:
    """Signal Engine for managing and distributing trading signals"""
    
    def __init__(self):
        self.is_running_flag = False
        self.signal_callbacks: List[Callable] = []
        self.execution_callbacks: List[Callable] = []
        
        # Signal storage
        self.active_signals: Dict[str, TradingSignal] = {}
        self.signal_history: List[TradingSignal] = []
        self.max_history_size = 1000
        
        # Database
        self.db_path = "signals.db"
        self._init_database()
        
        # Configuration
        self.signal_expiry_hours = 24
        self.min_confidence_threshold = 0.7
        self.auto_execution = False  # Set to True for live trading
        
        # Statistics
        self.stats = {
            "total_signals": 0,
            "executed_signals": 0,
            "cancelled_signals": 0,
            "expired_signals": 0,
            "total_pnl": 0.0
        }
        
        logger.info("Signal Engine initialized")
    
    def _init_database(self):
        """Initialize SQLite database for signal storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    divergence_type TEXT NOT NULL,
                    confidence_level REAL NOT NULL,
                    price REAL NOT NULL,
                    status TEXT NOT NULL,
                    indicators TEXT NOT NULL,
                    description TEXT,
                    execution_price REAL,
                    execution_time TEXT,
                    pnl REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create signal_history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_history (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    divergence_type TEXT NOT NULL,
                    confidence_level REAL NOT NULL,
                    price REAL NOT NULL,
                    status TEXT NOT NULL,
                    indicators TEXT NOT NULL,
                    description TEXT,
                    execution_price REAL,
                    execution_time TEXT,
                    pnl REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON signals(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON signals(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON signals(status)')
            
            conn.commit()
            conn.close()
            
            logger.info("Signal database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    async def start(self):
        """Start the signal engine"""
        if self.is_running_flag:
            return
        
        self.is_running_flag = True
        logger.info("Starting Signal Engine...")
        
        # Start background tasks
        asyncio.create_task(self._run_signal_monitor())
        asyncio.create_task(self._run_cleanup_task())
        
        logger.info("Signal Engine started")
    
    async def stop(self):
        """Stop the signal engine"""
        self.is_running_flag = False
        logger.info("Stopping Signal Engine...")
        
        # Save all active signals to history
        await self._save_all_signals()
        
        logger.info("Signal Engine stopped")
    
    def is_running(self) -> bool:
        """Check if the engine is running"""
        return self.is_running_flag
    
    async def _run_signal_monitor(self):
        """Monitor active signals and handle status changes"""
        while self.is_running_flag:
            try:
                # Check for expired signals
                await self._check_expired_signals()
                
                # Update signal statistics
                await self._update_statistics()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Signal monitor error: {e}")
                await asyncio.sleep(10)
    
    async def _run_cleanup_task(self):
        """Clean up old signals and maintain database"""
        while self.is_running_flag:
            try:
                # Clean up old history
                await self._cleanup_old_history()
                
                # Optimize database
                await self._optimize_database()
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(300)
    
    async def _check_expired_signals(self):
        """Check and expire old signals"""
        current_time = datetime.now()
        expired_signals = []
        
        for signal_id, signal in self.active_signals.items():
            if signal.status == SignalStatus.PENDING:
                # Check if signal has expired
                if (current_time - signal.timestamp).total_seconds() > (self.signal_expiry_hours * 3600):
                    signal.status = SignalStatus.EXPIRED
                    expired_signals.append(signal_id)
                    self.stats["expired_signals"] += 1
        
        # Remove expired signals from active list
        for signal_id in expired_signals:
            expired_signal = self.active_signals.pop(signal_id)
            self.signal_history.append(expired_signal)
            
            # Save to database
            await self._save_signal_to_db(expired_signal)
            
            logger.info(f"Signal {signal_id} expired and moved to history")
    
    async def _update_statistics(self):
        """Update signal statistics"""
        try:
            # Count active signals by status
            status_counts = {}
            for signal in self.active_signals.values():
                status = signal.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Update stats
            self.stats["active_signals"] = len(self.active_signals)
            self.stats["pending_signals"] = status_counts.get("pending", 0)
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    async def _cleanup_old_history(self):
        """Clean up old signal history"""
        try:
            # Keep only recent history in memory
            if len(self.signal_history) > self.max_history_size:
                old_signals = self.signal_history[:-self.max_history_size]
                self.signal_history = self.signal_history[-self.max_history_size:]
                
                logger.info(f"Cleaned up {len(old_signals)} old signals from memory")
                
        except Exception as e:
            logger.error(f"Error cleaning up history: {e}")
    
    async def _optimize_database(self):
        """Optimize SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Analyze tables
            cursor.execute("ANALYZE")
            
            # Vacuum database
            cursor.execute("VACUUM")
            
            conn.commit()
            conn.close()
            
            logger.debug("Database optimization completed")
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
    
    async def process_divergence_signal(self, divergence_signal: Any) -> Optional[TradingSignal]:
        """Process a divergence signal and create a trading signal"""
        try:
            # Check confidence threshold
            if divergence_signal.confidence_level < self.min_confidence_threshold:
                logger.debug(f"Signal confidence {divergence_signal.confidence_level} below threshold {self.min_confidence_threshold}")
                return None
            
            # Generate unique signal ID
            signal_id = self._generate_signal_id(divergence_signal)
            
            # Check if signal already exists
            if signal_id in self.active_signals:
                logger.debug(f"Signal {signal_id} already exists")
                return self.active_signals[signal_id]
            
            # Create trading signal
            trading_signal = TradingSignal(
                id=signal_id,
                timestamp=divergence_signal.timestamp,
                symbol=divergence_signal.symbol,
                signal_type=divergence_signal.signal_type.name,
                divergence_type=divergence_signal.divergence_type.value,
                confidence_level=divergence_signal.confidence_level,
                price=divergence_signal.price,
                status=SignalStatus.PENDING,
                indicators=divergence_signal.indicators,
                description=divergence_signal.description
            )
            
            # Store signal
            self.active_signals[signal_id] = trading_signal
            self.signal_history.append(trading_signal)
            self.stats["total_signals"] += 1
            
            # Save to database
            await self._save_signal_to_db(trading_signal)
            
            # Notify callbacks
            await self._notify_signal_callbacks(trading_signal)
            
            logger.info(f"Created trading signal {signal_id} for {divergence_signal.symbol}")
            
            # Auto-execute if enabled
            if self.auto_execution:
                await self.execute_signal(signal_id)
            
            return trading_signal
            
        except Exception as e:
            logger.error(f"Error processing divergence signal: {e}")
            return None
    
    def _generate_signal_id(self, divergence_signal: Any) -> str:
        """Generate unique signal ID"""
        # Create hash from signal components
        signal_string = f"{divergence_signal.symbol}_{divergence_signal.timestamp.isoformat()}_{divergence_signal.divergence_type.value}_{divergence_signal.signal_type.name}"
        return hashlib.md5(signal_string.encode()).hexdigest()[:16]
    
    async def execute_signal(self, signal_id: str) -> bool:
        """Execute a trading signal"""
        try:
            if signal_id not in self.active_signals:
                logger.warning(f"Signal {signal_id} not found")
                return False
            
            signal = self.active_signals[signal_id]
            
            if signal.status != SignalStatus.PENDING:
                logger.warning(f"Signal {signal_id} is not pending (status: {signal.status.value})")
                return False
            
            # Update signal status
            signal.status = SignalStatus.EXECUTED
            signal.execution_time = datetime.now()
            signal.execution_price = signal.price  # In real implementation, get actual execution price
            
            # Update statistics
            self.stats["executed_signals"] += 1
            
            # Save to database
            await self._save_signal_to_db(signal)
            
            # Notify execution callbacks
            await self._notify_execution_callbacks(signal)
            
            logger.info(f"Signal {signal_id} executed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing signal {signal_id}: {e}")
            return False
    
    async def cancel_signal(self, signal_id: str) -> bool:
        """Cancel a trading signal"""
        try:
            if signal_id not in self.active_signals:
                logger.warning(f"Signal {signal_id} not found")
                return False
            
            signal = self.active_signals[signal_id]
            
            if signal.status != SignalStatus.PENDING:
                logger.warning(f"Signal {signal_id} cannot be cancelled (status: {signal.status.value})")
                return False
            
            # Update signal status
            signal.status = SignalStatus.CANCELLED
            
            # Update statistics
            self.stats["cancelled_signals"] += 1
            
            # Save to database
            await self._save_signal_to_db(signal)
            
            logger.info(f"Signal {signal_id} cancelled successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling signal {signal_id}: {e}")
            return False
    
    async def _save_signal_to_db(self, signal: TradingSignal):
        """Save signal to SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert or update signal
            cursor.execute('''
                INSERT OR REPLACE INTO signals 
                (id, timestamp, symbol, signal_type, divergence_type, confidence_level, 
                 price, status, indicators, description, execution_price, execution_time, pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.id,
                signal.timestamp.isoformat(),
                signal.symbol,
                signal.signal_type,
                signal.divergence_type,
                signal.confidence_level,
                signal.price,
                signal.status.value,
                json.dumps(signal.indicators),
                signal.description,
                signal.execution_price,
                signal.execution_time.isoformat() if signal.execution_time else None,
                signal.pnl
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save signal to database: {e}")
    
    async def _save_all_signals(self):
        """Save all active signals to database"""
        for signal in self.active_signals.values():
            await self._save_signal_to_db(signal)
    
    async def _notify_signal_callbacks(self, signal: TradingSignal):
        """Notify all registered signal callbacks"""
        for callback in self.signal_callbacks:
            try:
                await callback(signal)
            except Exception as e:
                logger.error(f"Signal callback notification failed: {e}")
    
    async def _notify_execution_callbacks(self, signal: TradingSignal):
        """Notify all registered execution callbacks"""
        for callback in self.execution_callbacks:
            try:
                await callback(signal)
            except Exception as e:
                logger.error(f"Execution callback notification failed: {e}")
    
    def register_signal_callback(self, callback: Callable):
        """Register a callback for signal updates"""
        if callback not in self.signal_callbacks:
            self.signal_callbacks.append(callback)
            logger.info(f"Signal callback registered. Total callbacks: {len(self.signal_callbacks)}")
    
    def register_execution_callback(self, callback: Callable):
        """Register a callback for signal executions"""
        if callback not in self.execution_callbacks:
            self.execution_callbacks.append(callback)
            logger.info(f"Execution callback registered. Total callbacks: {len(self.execution_callbacks)}")
    
    def unregister_signal_callback(self, callback: Callable):
        """Unregister a signal callback"""
        if callback in self.signal_callbacks:
            self.signal_callbacks.remove(callback)
    
    def unregister_execution_callback(self, callback: Callable):
        """Unregister an execution callback"""
        if callback in self.execution_callbacks:
            self.execution_callbacks.remove(callback)
    
    def get_recent_signals(self, limit: int = 50) -> List[Dict]:
        """Get recent signals for API"""
        recent_signals = []
        
        # Get active signals
        for signal in list(self.active_signals.values())[-limit//2:]:
            recent_signals.append(asdict(signal))
        
        # Get recent history
        for signal in list(self.signal_history)[-limit//2:]:
            recent_signals.append(asdict(signal))
        
        # Sort by timestamp and limit
        recent_signals.sort(key=lambda x: x["timestamp"], reverse=True)
        return recent_signals[:limit]
    
    def get_signal_by_id(self, signal_id: str) -> Optional[Dict]:
        """Get signal by ID"""
        if signal_id in self.active_signals:
            return asdict(self.active_signals[signal_id])
        
        # Search in history
        for signal in self.signal_history:
            if signal.id == signal_id:
                return asdict(signal)
        
        return None
    
    def get_signals_by_symbol(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get signals for a specific symbol"""
        signals = []
        
        # Search active signals
        for signal in self.active_signals.values():
            if signal.symbol == symbol:
                signals.append(asdict(signal))
        
        # Search history
        for signal in self.signal_history:
            if signal.symbol == symbol:
                signals.append(asdict(signal))
        
        # Sort by timestamp and limit
        signals.sort(key=lambda x: x["timestamp"], reverse=True)
        return signals[:limit]
    
    def get_statistics(self) -> Dict:
        """Get signal engine statistics"""
        return {
            **self.stats,
            "active_signals": len(self.active_signals),
            "history_size": len(self.signal_history),
            "database_path": self.db_path
        }
    
    async def update_config(self, config: dict):
        """Update configuration"""
        if "signal_expiry_hours" in config:
            self.signal_expiry_hours = config["signal_expiry_hours"]
        if "min_confidence_threshold" in config:
            self.min_confidence_threshold = config["min_confidence_threshold"]
        if "auto_execution" in config:
            self.auto_execution = config["auto_execution"]
        if "max_history_size" in config:
            self.max_history_size = config["max_history_size"]
        
        logger.info("Signal engine configuration updated")
#!/usr/bin/env python3
"""
Real-Time Signal Pipeline
Integrates data fetching, AI signal generation, and alerts with sub-2-second latency
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import time
from collections import deque

from .ai_signal_generator import AISignalGenerator, Signal
from .twelve_data_client import TwelveDataClient, create_twelve_data_client
from .telegram_alerts import TelegramAlerts, create_telegram_alerts

logger = logging.getLogger(__name__)

@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    total_signals: int
    avg_latency_ms: float
    signals_per_hour: float
    last_signal_time: Optional[datetime]
    uptime_seconds: float
    data_feed_status: str
    model_status: str

class RealTimeSignalPipeline:
    """Real-time Forex signal pipeline with AI integration"""
    
    def __init__(self, symbols: List[str] = None, update_interval: float = 1.0):
        self.symbols = symbols or ["EUR/USD"]
        self.update_interval = update_interval
        self.is_running = False
        
        # Components
        self.data_client: Optional[TwelveDataClient] = None
        self.ai_generator: Optional[AISignalGenerator] = None
        self.telegram_alerts: Optional[TelegramAlerts] = None
        
        # Data management
        self.market_data_cache = {}
        self.signal_history = deque(maxlen=1000)
        self.performance_metrics = {}
        
        # Pipeline metrics
        self.start_time = None
        self.total_signals = 0
        self.latency_history = deque(maxlen=100)
        
        # Callbacks
        self.signal_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # Configuration
        self.min_data_points = 100  # Minimum data points for signal generation
        self.signal_cooldown = 60  # Seconds between signals for same symbol
        self.last_signal_times = {}
        
        # Performance tracking
        self.hourly_signals = deque(maxlen=3600)  # Track signals per hour
        
    async def initialize(self):
        """Initialize all pipeline components"""
        try:
            logger.info("Initializing Real-Time Signal Pipeline...")
            
            # Initialize Twelve Data client
            try:
                self.data_client = create_twelve_data_client()
                logger.info("Twelve Data client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Twelve Data client: {e}")
                raise
            
            # Initialize AI signal generator
            try:
                self.ai_generator = AISignalGenerator()
                logger.info("AI Signal Generator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize AI Signal Generator: {e}")
                raise
            
            # Initialize Telegram alerts (optional)
            try:
                self.telegram_alerts = create_telegram_alerts()
                if self.telegram_alerts:
                    logger.info("Telegram alerts initialized")
                else:
                    logger.info("Telegram alerts not configured")
            except Exception as e:
                logger.warning(f"Failed to initialize Telegram alerts: {e}")
                self.telegram_alerts = None
            
            # Test connections
            await self._test_connections()
            
            # Initialize market data cache
            await self._initialize_market_data()
            
            logger.info("Real-Time Signal Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    async def _test_connections(self):
        """Test all external connections"""
        try:
            # Test Twelve Data connection
            if self.data_client:
                # Test with a simple API call
                test_data = await self.data_client.get_real_time_quote("EUR/USD")
                logger.info("Twelve Data connection test successful")
            
            # Test Telegram connection
            if self.telegram_alerts:
                telegram_ok = await self.telegram_alerts.test_connection()
                if telegram_ok:
                    logger.info("Telegram connection test successful")
                else:
                    logger.warning("Telegram connection test failed")
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            raise
    
    async def _initialize_market_data(self):
        """Initialize market data cache with historical data"""
        try:
            logger.info("Initializing market data cache...")
            
            for symbol in self.symbols:
                try:
                    # Fetch historical data for training and initial cache
                    historical_data = await self.data_client.get_historical_data(
                        symbol=symbol,
                        interval="1min",
                        outputsize=2000  # 2000 minutes of data
                    )
                    
                    # Store in cache
                    self.market_data_cache[symbol] = historical_data
                    
                    logger.info(f"Initialized data cache for {symbol}: {len(historical_data)} records")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize data for {symbol}: {e}")
                    # Create empty DataFrame as fallback
                    self.market_data_cache[symbol] = pd.DataFrame()
            
            # Train AI model if we have sufficient data
            await self._train_ai_model()
            
        except Exception as e:
            logger.error(f"Failed to initialize market data: {e}")
            raise
    
    async def _train_ai_model(self):
        """Train the AI model with historical data"""
        try:
            logger.info("Training AI model with historical data...")
            
            # Combine data from all symbols for training
            all_data = []
            for symbol, data in self.market_data_cache.items():
                if len(data) > 0:
                    all_data.append(data)
            
            if len(all_data) == 0:
                logger.warning("No historical data available for training")
                return
            
            # Combine and prepare training data
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
            
            # Ensure we have enough data
            if len(combined_data) < self.min_data_points:
                logger.warning(f"Insufficient data for training: {len(combined_data)} < {self.min_data_points}")
                return
            
            # Train the model
            training_metrics = await self.ai_generator.train_model(combined_data)
            
            logger.info(f"AI model training completed: {training_metrics.accuracy:.2%} accuracy")
            
            # Send performance alert
            if self.telegram_alerts:
                await self.telegram_alerts.send_performance_alert({
                    'accuracy': training_metrics.accuracy,
                    'sharpe_ratio': training_metrics.sharpe_ratio,
                    'win_rate': training_metrics.win_rate,
                    'max_drawdown': training_metrics.max_drawdown,
                    'total_trades': training_metrics.total_trades,
                    'last_signal_action': 'None',
                    'live_signals_count': 0
                })
            
        except Exception as e:
            logger.error(f"Failed to train AI model: {e}")
            if self.telegram_alerts:
                await self.telegram_alerts.send_error_alert(f"AI model training failed: {e}")
    
    async def start(self):
        """Start the real-time signal pipeline"""
        try:
            if self.is_running:
                logger.warning("Pipeline is already running")
                return
            
            logger.info("Starting Real-Time Signal Pipeline...")
            
            # Initialize if not already done
            if not self.data_client or not self.ai_generator:
                await self.initialize()
            
            # Start components
            if self.ai_generator:
                await self.ai_generator.start()
            
            if self.telegram_alerts:
                await self.telegram_alerts.start()
            
            # Start the main pipeline loop
            self.is_running = True
            self.start_time = datetime.utcnow()
            
            # Start data update loop
            asyncio.create_task(self._data_update_loop())
            
            # Start signal generation loop
            asyncio.create_task(self._signal_generation_loop())
            
            logger.info("Real-Time Signal Pipeline started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            raise
    
    async def stop(self):
        """Stop the real-time signal pipeline"""
        try:
            logger.info("Stopping Real-Time Signal Pipeline...")
            
            self.is_running = False
            
            # Stop components
            if self.ai_generator:
                await self.ai_generator.stop()
            
            if self.telegram_alerts:
                await self.telegram_alerts.stop()
            
            logger.info("Real-Time Signal Pipeline stopped")
            
        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")
    
    async def _data_update_loop(self):
        """Continuous loop for updating market data"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Update data for all symbols
                for symbol in self.symbols:
                    try:
                        await self._update_symbol_data(symbol)
                    except Exception as e:
                        logger.error(f"Error updating data for {symbol}: {e}")
                        if self.telegram_alerts:
                            await self.telegram_alerts.send_error_alert(f"Data update failed for {symbol}: {e}")
                
                # Calculate update time
                update_time = (time.time() - start_time) * 1000
                if update_time > 1000:  # Log if update takes more than 1 second
                    logger.warning(f"Data update took {update_time:.2f}ms")
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in data update loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _update_symbol_data(self, symbol: str):
        """Update market data for a specific symbol"""
        try:
            # Get real-time quote
            quote = await self.data_client.get_real_time_quote(symbol)
            
            if quote and 'data' in quote:
                quote_data = quote['data']
                
                # Create new data point
                new_data = {
                    'timestamp': datetime.fromtimestamp(int(quote_data.get('timestamp', 0)) / 1000),
                    'open': float(quote_data.get('open', 0)),
                    'high': float(quote_data.get('high', 0)),
                    'low': float(quote_data.get('low', 0)),
                    'close': float(quote_data.get('close', 0)),
                    'volume': float(quote_data.get('volume', 0)),
                    'symbol': symbol
                }
                
                # Add to cache
                if symbol in self.market_data_cache:
                    # Append new data point
                    new_df = pd.DataFrame([new_data])
                    self.market_data_cache[symbol] = pd.concat([
                        self.market_data_cache[symbol], new_df
                    ], ignore_index=True)
                    
                    # Keep only recent data (last 2000 points)
                    if len(self.market_data_cache[symbol]) > 2000:
                        self.market_data_cache[symbol] = self.market_data_cache[symbol].tail(2000)
                    
                    # Remove duplicates
                    self.market_data_cache[symbol] = self.market_data_cache[symbol].drop_duplicates(subset=['timestamp'])
                    self.market_data_cache[symbol] = self.market_data_cache[symbol].sort_values('timestamp').reset_index(drop=True)
                
                else:
                    # Initialize cache with new data
                    self.market_data_cache[symbol] = pd.DataFrame([new_data])
                
        except Exception as e:
            logger.error(f"Error updating data for {symbol}: {e}")
            raise
    
    async def _signal_generation_loop(self):
        """Continuous loop for generating trading signals"""
        while self.is_running:
            try:
                # Generate signals for all symbols
                for symbol in self.symbols:
                    try:
                        await self._generate_symbol_signal(symbol)
                    except Exception as e:
                        logger.error(f"Error generating signal for {symbol}: {e}")
                        if self.telegram_alerts:
                            await self.telegram_alerts.send_error_alert(f"Signal generation failed for {symbol}: {e}")
                
                # Wait before next signal generation cycle
                await asyncio.sleep(10)  # Check for signals every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in signal generation loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying
    
    async def _generate_symbol_signal(self, symbol: str):
        """Generate trading signal for a specific symbol"""
        try:
            # Check cooldown
            if symbol in self.last_signal_times:
                time_since_last = (datetime.utcnow() - self.last_signal_times[symbol]).total_seconds()
                if time_since_last < self.signal_cooldown:
                    return
            
            # Get market data
            if symbol not in self.market_data_cache or len(self.market_data_cache[symbol]) < self.min_data_points:
                return
            
            market_data = self.market_data_cache[symbol]
            
            # Generate signal
            start_time = time.time()
            signal = await self.ai_generator.generate_signal(market_data)
            generation_time = (time.time() - start_time) * 1000
            
            if signal:
                # Record signal
                self._record_signal(signal, generation_time)
                
                # Update last signal time
                self.last_signal_times[symbol] = datetime.utcnow()
                
                # Send alerts
                await self._send_signal_alerts(signal)
                
                # Notify callbacks
                await self._notify_signal_callbacks(signal)
                
                logger.info(f"Signal generated for {symbol}: {signal.action} (confidence: {signal.confidence:.2%})")
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            raise
    
    def _record_signal(self, signal: Signal, generation_time: float):
        """Record signal and update metrics"""
        try:
            # Add to history
            self.signal_history.append(signal)
            
            # Update metrics
            self.total_signals += 1
            self.latency_history.append(generation_time)
            self.hourly_signals.append(datetime.utcnow())
            
            # Clean up old hourly signals
            current_time = datetime.utcnow()
            while self.hourly_signals and (current_time - self.hourly_signals[0]).total_seconds() > 3600:
                self.hourly_signals.popleft()
            
        except Exception as e:
            logger.error(f"Error recording signal: {e}")
    
    async def _send_signal_alerts(self, signal: Signal):
        """Send signal alerts via Telegram"""
        try:
            if self.telegram_alerts:
                # Convert signal to dict for alert
                signal_dict = {
                    'symbol': signal.symbol,
                    'action': signal.action,
                    'confidence': signal.confidence,
                    'price': signal.price,
                    'timestamp': signal.timestamp.isoformat(),
                    'model_version': signal.model_version
                }
                
                await self.telegram_alerts.send_signal_alert(signal_dict)
                
        except Exception as e:
            logger.error(f"Error sending signal alerts: {e}")
    
    async def _notify_signal_callbacks(self, signal: Signal):
        """Notify registered signal callbacks"""
        try:
            for callback in self.signal_callbacks:
                try:
                    await callback(signal)
                except Exception as e:
                    logger.error(f"Error in signal callback: {e}")
        except Exception as e:
            logger.error(f"Error notifying signal callbacks: {e}")
    
    def add_signal_callback(self, callback: Callable):
        """Add callback for signal notifications"""
        self.signal_callbacks.append(callback)
    
    def remove_signal_callback(self, callback: Callable):
        """Remove signal callback"""
        if callback in self.signal_callbacks:
            self.signal_callbacks.remove(callback)
    
    def add_error_callback(self, callback: Callable):
        """Add callback for error notifications"""
        self.error_callbacks.append(callback)
    
    def remove_error_callback(self, callback: Callable):
        """Remove error callback"""
        if callback in self.error_callbacks:
            self.error_callbacks.remove(callback)
    
    def get_pipeline_metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics"""
        try:
            current_time = datetime.utcnow()
            
            # Calculate uptime
            uptime = 0
            if self.start_time:
                uptime = (current_time - self.start_time).total_seconds()
            
            # Calculate average latency
            avg_latency = 0
            if self.latency_history:
                avg_latency = np.mean(self.latency_history)
            
            # Calculate signals per hour
            signals_per_hour = len(self.hourly_signals)
            
            # Get last signal time
            last_signal_time = None
            if self.signal_history:
                last_signal_time = self.signal_history[-1].timestamp
            
            # Get component statuses
            data_feed_status = "connected" if self.data_client and self.data_client.is_connected else "disconnected"
            model_status = "running" if self.ai_generator and self.ai_generator.is_running else "stopped"
            
            return PipelineMetrics(
                total_signals=self.total_signals,
                avg_latency_ms=avg_latency,
                signals_per_hour=signals_per_hour,
                last_signal_time=last_signal_time,
                uptime_seconds=uptime,
                data_feed_status=data_feed_status,
                model_status=model_status
            )
            
        except Exception as e:
            logger.error(f"Error getting pipeline metrics: {e}")
            return PipelineMetrics(
                total_signals=0,
                avg_latency_ms=0,
                signals_per_hour=0,
                last_signal_time=None,
                uptime_seconds=0,
                data_feed_status="unknown",
                model_status="unknown"
            )
    
    def get_recent_signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trading signals"""
        try:
            recent_signals = []
            for signal in list(self.signal_history)[-limit:]:
                signal_dict = {
                    'symbol': signal.symbol,
                    'action': signal.action,
                    'confidence': signal.confidence,
                    'price': signal.price,
                    'timestamp': signal.timestamp.isoformat(),
                    'model_version': signal.model_version
                }
                recent_signals.append(signal_dict)
            
            return recent_signals
            
        except Exception as e:
            logger.error(f"Error getting recent signals: {e}")
            return []
    
    def get_market_data_summary(self) -> Dict[str, Any]:
        """Get summary of current market data"""
        try:
            summary = {}
            for symbol, data in self.market_data_cache.items():
                if len(data) > 0:
                    latest = data.iloc[-1]
                    summary[symbol] = {
                        'last_price': latest['close'],
                        'last_update': latest['timestamp'].isoformat(),
                        'data_points': len(data),
                        'price_change_1h': self._calculate_price_change(data, hours=1),
                        'price_change_24h': self._calculate_price_change(data, hours=24)
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting market data summary: {e}")
            return {}
    
    def _calculate_price_change(self, data: pd.DataFrame, hours: int) -> float:
        """Calculate price change over specified hours"""
        try:
            if len(data) < 2:
                return 0.0
            
            current_time = data.iloc[-1]['timestamp']
            target_time = current_time - timedelta(hours=hours)
            
            # Find closest data point to target time
            time_diff = abs(data['timestamp'] - target_time)
            closest_idx = time_diff.idxmin()
            
            if closest_idx < len(data) - 1:
                old_price = data.iloc[closest_idx]['close']
                new_price = data.iloc[-1]['close']
                return ((new_price - old_price) / old_price) * 100
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating price change: {e}")
            return 0.0
    
    async def retrain_model(self):
        """Retrain the AI model with latest data"""
        try:
            logger.info("Starting model retraining...")
            
            # Stop signal generation temporarily
            was_running = self.is_running
            if was_running:
                await self.stop()
            
            # Retrain model
            await self._train_ai_model()
            
            # Restart if it was running
            if was_running:
                await self.start()
            
            logger.info("Model retraining completed")
            
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            if self.telegram_alerts:
                await self.telegram_alerts.send_error_alert(f"Model retraining failed: {e}")
            raise
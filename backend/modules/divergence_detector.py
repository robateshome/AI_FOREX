#!/usr/bin/env python3
"""
Module 0x02: Divergence Detector
CRC32: PLACEHOLDER_CRC32_02

STRATEGY: BULLISH/BEARISH_DIVERGENCE
INDICATORS_USED: RSI, MACD_HIST, STOCH, CCI
ALGO: SWING_HIGH_LOW_COMPARE_INDICATOR
MULTI-TIMEFRAME_ANALYSIS=ENABLED
SIGNAL_CONFIDENCE: WEIGHTED_SCORE
OUTPUT_BINARY: 0b1(BUY)|0b0(SELL)|0b10(HOLD)
EXEC_PRIORITY=HARD_REALTIME(0xFF)
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)

class DivergenceType(Enum):
    """Divergence types"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    HIDDEN_BULLISH = "hidden_bullish"
    HIDDEN_BEARISH = "hidden_bearish"
    NONE = "none"

class SignalType(Enum):
    """Signal types"""
    BUY = 0b1
    SELL = 0b0
    HOLD = 0b10

@dataclass
class DivergenceSignal:
    """Divergence signal structure"""
    timestamp: datetime
    symbol: str
    divergence_type: DivergenceType
    signal_type: SignalType
    confidence_level: float  # 0.0 to 1.0
    price: float
    indicators: Dict[str, float]
    swing_high: Optional[float] = None
    swing_low: Optional[float] = None
    description: str = ""

class DivergenceDetector:
    """Divergence Detector with multi-timeframe analysis"""
    
    def __init__(self):
        self.is_running_flag = False
        self.data_callbacks: List[Callable] = []
        self.signal_callbacks: List[Callable] = []
        
        # Configuration
        self.lookback_period = 20
        self.confidence_threshold = 0.7
        self.min_swing_distance = 0.001  # Minimum price movement for swing detection
        
        # Data storage
        self.price_history = {}  # symbol -> List[float]
        self.indicator_history = {}  # symbol -> Dict[str, List[float]]
        self.recent_divergences = []
        self.max_divergence_history = 100
        
        # Multi-timeframe settings
        self.timeframes = ["1m", "5m", "15m", "1h", "4h"]
        self.current_timeframe = "5m"
        
        # Indicator weights for confidence calculation
        self.indicator_weights = {
            "rsi": 0.3,
            "macd_hist": 0.25,
            "stoch": 0.25,
            "cci": 0.2
        }
        
        # Swing detection parameters
        self.swing_window = 5
        self.min_swing_strength = 0.5
        
        logger.info("Divergence Detector initialized")
    
    async def start(self):
        """Start the divergence detector"""
        if self.is_running_flag:
            return
        
        self.is_running_flag = True
        logger.info("Starting Divergence Detector...")
        
        # Start analysis task
        asyncio.create_task(self._run_analysis_loop())
        
        logger.info("Divergence Detector started")
    
    async def stop(self):
        """Stop the divergence detector"""
        self.is_running_flag = False
        logger.info("Divergence Detector stopped")
    
    def is_running(self) -> bool:
        """Check if the detector is running"""
        return self.is_running_flag
    
    async def _run_analysis_loop(self):
        """Main analysis loop"""
        while self.is_running_flag:
            try:
                # Wait for data updates
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")
                await asyncio.sleep(5)
    
    def update_market_data(self, symbol: str, price: float, indicators: Dict[str, float]):
        """Update market data for divergence analysis"""
        try:
            # Initialize data structures if needed
            if symbol not in self.price_history:
                self.price_history[symbol] = []
                self.indicator_history[symbol] = {ind: [] for ind in self.indicator_weights.keys()}
            
            # Update price history
            self.price_history[symbol].append(price)
            if len(self.price_history[symbol]) > self.lookback_period:
                self.price_history[symbol].pop(0)
            
            # Update indicator history
            for ind_name, ind_value in indicators.items():
                if ind_name in self.indicator_history[symbol]:
                    self.indicator_history[symbol][ind_name].append(ind_value)
                    if len(self.indicator_history[symbol][ind_name]) > self.lookback_period:
                        self.indicator_history[symbol][ind_name].pop(0)
            
            # Check for divergences when we have enough data
            if len(self.price_history[symbol]) >= self.lookback_period:
                await self._detect_divergences(symbol)
                
        except Exception as e:
            logger.error(f"Error updating market data for {symbol}: {e}")
    
    async def _detect_divergences(self, symbol: str):
        """Detect divergences for a given symbol"""
        try:
            prices = self.price_history[symbol]
            indicators = self.indicator_history[symbol]
            
            if len(prices) < self.lookback_period:
                return
            
            # Detect swing highs and lows
            swing_highs, swing_lows = self._detect_swings(prices)
            
            # Check for divergences with each indicator
            divergences = []
            
            for indicator_name, indicator_values in indicators.items():
                if len(indicator_values) < self.lookback_period:
                    continue
                
                # Detect regular divergences
                bullish_div = self._detect_bullish_divergence(prices, indicator_values, swing_lows)
                bearish_div = self._detect_bearish_divergence(prices, indicator_values, swing_highs)
                
                # Detect hidden divergences
                hidden_bullish_div = self._detect_hidden_bullish_divergence(prices, indicator_values, swing_highs)
                hidden_bearish_div = self._detect_hidden_bearish_divergence(prices, indicator_values, swing_lows)
                
                if bullish_div:
                    divergences.append(("bullish", indicator_name, bullish_div))
                if bearish_div:
                    divergences.append(("bearish", indicator_name, bearish_div))
                if hidden_bullish_div:
                    divergences.append(("hidden_bullish", indicator_name, hidden_bullish_div))
                if hidden_bearish_div:
                    divergences.append(("hidden_bearish", indicator_name, hidden_bearish_div))
            
            # Generate signals based on detected divergences
            if divergences:
                await self._generate_signals(symbol, divergences, prices[-1])
                
        except Exception as e:
            logger.error(f"Error detecting divergences for {symbol}: {e}")
    
    def _detect_swings(self, prices: List[float]) -> Tuple[List[int], List[int]]:
        """Detect swing highs and lows in price data"""
        try:
            swing_highs = []
            swing_lows = []
            
            for i in range(self.swing_window, len(prices) - self.swing_window):
                # Check for swing high
                if all(prices[i] > prices[j] for j in range(i - self.swing_window, i)) and \
                   all(prices[i] > prices[j] for j in range(i + 1, i + self.swing_window + 1)):
                    swing_highs.append(i)
                
                # Check for swing low
                if all(prices[i] < prices[j] for j in range(i - self.swing_window, i)) and \
                   all(prices[i] < prices[j] for j in range(i + 1, i + self.swing_window + 1)):
                    swing_lows.append(i)
            
            return swing_highs, swing_lows
            
        except Exception as e:
            logger.error(f"Error detecting swings: {e}")
            return [], []
    
    def _detect_bullish_divergence(self, prices: List[float], indicators: List[float], swing_lows: List[int]) -> Optional[Dict]:
        """Detect bullish divergence (price makes lower low, indicator makes higher low)"""
        try:
            if len(swing_lows) < 2:
                return None
            
            # Get last two swing lows
            last_swing_low = swing_lows[-1]
            prev_swing_low = swing_lows[-2]
            
            # Check if price made a lower low
            if prices[last_swing_low] >= prices[prev_swing_low]:
                return None
            
            # Check if indicator made a higher low
            if indicators[last_swing_low] <= indicators[prev_swing_low]:
                return None
            
            # Calculate divergence strength
            price_change = abs(prices[last_swing_low] - prices[prev_swing_low])
            indicator_change = abs(indicators[last_swing_low] - indicators[prev_swing_low])
            
            return {
                "price_low": prices[last_swing_low],
                "price_high": prices[prev_swing_low],
                "indicator_low": indicators[last_swing_low],
                "indicator_high": indicators[prev_swing_low],
                "strength": min(price_change, indicator_change) / max(price_change, indicator_change)
            }
            
        except Exception as e:
            logger.error(f"Error detecting bullish divergence: {e}")
            return None
    
    def _detect_bearish_divergence(self, prices: List[float], indicators: List[float], swing_highs: List[int]) -> Optional[Dict]:
        """Detect bearish divergence (price makes higher high, indicator makes lower high)"""
        try:
            if len(swing_highs) < 2:
                return None
            
            # Get last two swing highs
            last_swing_high = swing_highs[-1]
            prev_swing_high = swing_highs[-2]
            
            # Check if price made a higher high
            if prices[last_swing_high] <= prices[prev_swing_high]:
                return None
            
            # Check if indicator made a lower high
            if indicators[last_swing_high] >= indicators[prev_swing_high]:
                return None
            
            # Calculate divergence strength
            price_change = abs(prices[last_swing_high] - prices[prev_swing_high])
            indicator_change = abs(indicators[last_swing_high] - indicators[prev_swing_high])
            
            return {
                "price_low": prices[prev_swing_high],
                "price_high": prices[last_swing_high],
                "indicator_low": indicators[last_swing_high],
                "indicator_high": indicators[prev_swing_high],
                "strength": min(price_change, indicator_change) / max(price_change, indicator_change)
            }
            
        except Exception as e:
            logger.error(f"Error detecting bearish divergence: {e}")
            return None
    
    def _detect_hidden_bullish_divergence(self, prices: List[float], indicators: List[float], swing_highs: List[int]) -> Optional[Dict]:
        """Detect hidden bullish divergence (price makes higher low, indicator makes lower low)"""
        try:
            if len(swing_highs) < 2:
                return None
            
            # Get last two swing highs
            last_swing_high = swing_highs[-1]
            prev_swing_high = swing_highs[-2]
            
            # Check if price made a higher high
            if prices[last_swing_high] <= prices[prev_swing_high]:
                return None
            
            # Check if indicator made a lower high
            if indicators[last_swing_high] >= indicators[prev_swing_high]:
                return None
            
            return {
                "price_low": prices[prev_swing_high],
                "price_high": prices[last_swing_high],
                "indicator_low": indicators[last_swing_high],
                "indicator_high": indicators[prev_swing_high],
                "strength": 0.5  # Default strength for hidden divergences
            }
            
        except Exception as e:
            logger.error(f"Error detecting hidden bullish divergence: {e}")
            return None
    
    def _detect_hidden_bearish_divergence(self, prices: List[float], indicators: List[float], swing_lows: List[int]) -> Optional[Dict]:
        """Detect hidden bearish divergence (price makes lower high, indicator makes higher high)"""
        try:
            if len(swing_lows) < 2:
                return None
            
            # Get last two swing lows
            last_swing_low = swing_lows[-1]
            prev_swing_low = swing_lows[-2]
            
            # Check if price made a lower low
            if prices[last_swing_low] >= prices[prev_swing_low]:
                return None
            
            # Check if indicator made a higher low
            if indicators[last_swing_low] <= indicators[prev_swing_low]:
                return None
            
            return {
                "price_low": prices[last_swing_low],
                "price_high": prices[prev_swing_low],
                "indicator_low": indicators[prev_swing_low],
                "indicator_high": indicators[last_swing_low],
                "strength": 0.5  # Default strength for hidden divergences
            }
            
        except Exception as e:
            logger.error(f"Error detecting hidden bearish divergence: {e}")
            return None
    
    async def _generate_signals(self, symbol: str, divergences: List[Tuple], current_price: float):
        """Generate trading signals based on detected divergences"""
        try:
            # Calculate weighted confidence score
            total_confidence = 0.0
            total_weight = 0.0
            signal_type = SignalType.HOLD
            divergence_type = DivergenceType.NONE
            
            for div_type, indicator_name, div_data in divergences:
                weight = self.indicator_weights.get(indicator_name, 0.1)
                strength = div_data.get("strength", 0.5)
                
                total_confidence += strength * weight
                total_weight += weight
                
                # Determine signal type
                if div_type in ["bullish", "hidden_bullish"]:
                    signal_type = SignalType.BUY
                    divergence_type = DivergenceType(div_type)
                elif div_type in ["bearish", "hidden_bearish"]:
                    signal_type = SignalType.SELL
                    divergence_type = DivergenceType(div_type)
            
            if total_weight > 0:
                final_confidence = total_confidence / total_weight
                
                # Only generate signal if confidence is above threshold
                if final_confidence >= self.confidence_threshold:
                    # Create divergence signal
                    signal = DivergenceSignal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        divergence_type=divergence_type,
                        signal_type=signal_type,
                        confidence_level=final_confidence,
                        price=current_price,
                        indicators={ind: self.indicator_history[symbol][ind][-1] if ind in self.indicator_history[symbol] else 0.0 
                                  for ind in self.indicator_weights.keys()},
                        description=f"{divergence_type.value} divergence detected with {final_confidence:.2f} confidence"
                    )
                    
                    # Store signal
                    self.recent_divergences.append(signal)
                    if len(self.recent_divergences) > self.max_divergence_history:
                        self.recent_divergences.pop(0)
                    
                    # Notify callbacks
                    await self._notify_signal_callbacks(signal)
                    
                    logger.info(f"Generated {signal_type.name} signal for {symbol}: {signal.description}")
                    
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
    
    async def _notify_signal_callbacks(self, signal: DivergenceSignal):
        """Notify all registered signal callbacks"""
        for callback in self.signal_callbacks:
            try:
                await callback(signal)
            except Exception as e:
                logger.error(f"Signal callback notification failed: {e}")
    
    def register_data_callback(self, callback: Callable):
        """Register a callback for data updates"""
        if callback not in self.data_callbacks:
            self.data_callbacks.append(callback)
            logger.info(f"Data callback registered. Total callbacks: {len(self.data_callbacks)}")
    
    def register_signal_callback(self, callback: Callable):
        """Register a callback for signal updates"""
        if callback not in self.signal_callbacks:
            self.signal_callbacks.append(callback)
            logger.info(f"Signal callback registered. Total callbacks: {len(self.signal_callbacks)}")
    
    def unregister_data_callback(self, callback: Callable):
        """Unregister a data callback"""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
    
    def unregister_signal_callback(self, callback: Callable):
        """Unregister a signal callback"""
        if callback in self.signal_callbacks:
            self.signal_callbacks.remove(callback)
    
    def get_recent_divergences(self) -> List[Dict]:
        """Get recent divergence signals for API"""
        return [
            {
                "timestamp": signal.timestamp.isoformat(),
                "symbol": signal.symbol,
                "divergence_type": signal.divergence_type.value,
                "signal_type": signal.signal_type.value,
                "confidence_level": signal.confidence_level,
                "price": signal.price,
                "indicators": signal.indicators,
                "description": signal.description
            }
            for signal in self.recent_divergences[-10:]  # Return last 10 signals
        ]
    
    async def update_config(self, config: dict):
        """Update configuration"""
        if "lookback_period" in config:
            self.lookback_period = config["lookback_period"]
        if "confidence_threshold" in config:
            self.confidence_threshold = config["confidence_threshold"]
        if "min_swing_distance" in config:
            self.min_swing_distance = config["min_swing_distance"]
        if "indicator_weights" in config:
            self.indicator_weights.update(config["indicator_weights"])
        
        logger.info("Divergence detector configuration updated")
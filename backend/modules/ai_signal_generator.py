#!/usr/bin/env python3
"""
AI Forex Signal Generator Module
Implements LSTM-based signal generation with real-time processing
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# ML imports
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

# Technical analysis
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

logger = logging.getLogger(__name__)

@dataclass
class Signal:
    """Trading signal structure"""
    symbol: str
    timestamp: datetime
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    price: float
    features: Dict[str, float]
    model_version: str

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int

class AISignalGenerator:
    """AI-powered Forex signal generator using LSTM"""
    
    def __init__(self, model_path: str = "models/forex_lstm.h5"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = []
        self.is_running = False
        
        # Signal filtering
        self.signal_history = []
        self.min_consecutive_signals = 2
        self.last_signal = None
        
        # Model configuration
        self.sequence_length = 60  # 60 minutes of historical data
        self.feature_window = 15  # Last 15 close prices as lag features
        
        # Performance tracking
        self.backtest_results = None
        self.live_performance = []
        
        # Initialize model
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load existing model or create a new one"""
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                logger.info(f"Loaded existing model from {self.model_path}")
                
                # Load scaler and encoder
                scaler_path = self.model_path.replace('.h5', '_scaler.pkl')
                encoder_path = self.model_path.replace('.h5', '_encoder.pkl')
                
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                if os.path.exists(encoder_path):
                    self.label_encoder = joblib.load(encoder_path)
                    
            else:
                self._create_new_model()
                logger.info("Created new LSTM model")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._create_new_model()
    
    def _create_new_model(self):
        """Create a new LSTM model"""
        try:
            # Model architecture
            self.model = keras.Sequential([
                keras.layers.LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self._get_feature_count())),
                keras.layers.Dropout(0.2),
                keras.layers.LSTM(64, return_sequences=False),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(3, activation='softmax')  # buy, sell, hold
            ])
            
            # Compile model
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Initialize scaler and encoder
            self.scaler = StandardScaler()
            self.label_encoder = LabelEncoder()
            
            logger.info("New LSTM model created successfully")
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
    
    def _get_feature_count(self) -> int:
        """Calculate total number of features"""
        # Technical indicators
        tech_features = 5  # SMA, EMA, RSI, MACD, ATR
        
        # Lag features
        lag_features = self.feature_window
        
        # OHLCV features
        ohlcv_features = 5
        
        return tech_features + lag_features + ohlcv_features
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer technical features from OHLCV data"""
        try:
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError("Missing required OHLCV columns")
            
            # Technical indicators
            df['sma_10'] = SMAIndicator(close=df['close'], window=10).sma_indicator()
            df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
            df['rsi_14'] = RSIIndicator(close=df['close'], window=14).rsi()
            
            # MACD
            macd = MACD(close=df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_hist'] = macd.macd_diff()
            
            # Volatility
            df['atr_14'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
            
            # Lag features (past close prices)
            for i in range(1, self.feature_window + 1):
                df[f'close_lag_{i}'] = df['close'].shift(i)
            
            # Log returns
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            
            # Normalize volume
            df['volume_normalized'] = df['volume'] / df['volume'].rolling(window=20).mean()
            
            # Drop NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            raise
    
    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM model"""
        try:
            # Select feature columns
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'symbol']]
            self.feature_columns = feature_cols
            
            # Create sequences
            X, y = [], []
            
            for i in range(self.sequence_length, len(df)):
                # Input sequence
                sequence = df[feature_cols].iloc[i-self.sequence_length:i].values
                X.append(sequence)
                
                # Target (next period's direction)
                current_price = df['close'].iloc[i]
                next_price = df['close'].iloc[i+1] if i+1 < len(df) else current_price
                
                if next_price > current_price * 1.001:  # 0.1% threshold
                    y.append(0)  # buy
                elif next_price < current_price * 0.999:
                    y.append(1)  # sell
                else:
                    y.append(2)  # hold
            
            X = np.array(X)
            y = np.array(y)
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=3)
            
            return X, y_categorical
            
        except Exception as e:
            logger.error(f"Error preparing sequences: {e}")
            raise
    
    async def train_model(self, historical_data: pd.DataFrame) -> ModelMetrics:
        """Train the LSTM model on historical data"""
        try:
            logger.info("Starting model training...")
            
            # Engineer features
            df_features = self.engineer_features(historical_data.copy())
            
            # Prepare sequences
            X, y = self.prepare_sequences(df_features)
            
            # Split data chronologically
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
            
            X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
            X_test_scaled = self.scaler.transform(X_test_reshaped)
            
            # Reshape back to 3D
            X_train_scaled = X_train_scaled.reshape(X_train.shape)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
            
            # Train model
            history = self.model.fit(
                X_train_scaled, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=1
            )
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_test_classes = np.argmax(y_test, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_classes, y_pred_classes)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test_classes, y_pred_classes, average='weighted')
            
            # Backtest on test set
            backtest_metrics = self._backtest_signals(X_test_scaled, y_test_classes, y_pred_classes)
            
            # Save model
            self._save_model()
            
            metrics = ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                sharpe_ratio=backtest_metrics['sharpe_ratio'],
                max_drawdown=backtest_metrics['max_drawdown'],
                win_rate=backtest_metrics['win_rate'],
                total_trades=backtest_metrics['total_trades']
            )
            
            self.backtest_results = metrics
            logger.info(f"Model training completed. Accuracy: {accuracy:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def _backtest_signals(self, X_test: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Backtest trading signals"""
        try:
            # Simulate trading
            initial_capital = 10000
            capital = initial_capital
            trades = []
            equity_curve = [initial_capital]
            
            for i in range(len(y_pred)):
                if y_pred[i] != 2:  # Not hold
                    # Simple position sizing (1% of capital)
                    position_size = capital * 0.01
                    
                    if y_pred[i] == 0:  # Buy
                        # Assume 0.1% profit/loss per trade
                        if y_true[i] == 0:  # Correct prediction
                            capital += position_size * 0.001
                        else:
                            capital -= position_size * 0.001
                    elif y_pred[i] == 1:  # Sell
                        if y_true[i] == 1:  # Correct prediction
                            capital += position_size * 0.001
                        else:
                            capital -= position_size * 0.001
                    
                    trades.append({
                        'prediction': y_pred[i],
                        'actual': y_true[i],
                        'capital': capital
                    })
                
                equity_curve.append(capital)
            
            # Calculate metrics
            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            # Max drawdown
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak
            max_drawdown = np.max(drawdown)
            
            # Win rate
            winning_trades = sum(1 for trade in trades if trade['capital'] > trade.get('prev_capital', initial_capital))
            win_rate = winning_trades / len(trades) if trades else 0
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': len(trades)
            }
            
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            return {
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0
            }
    
    def _save_model(self):
        """Save the trained model and scaler"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save model
            self.model.save(self.model_path)
            
            # Save scaler
            scaler_path = self.model_path.replace('.h5', '_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            
            # Save encoder
            encoder_path = self.model_path.replace('.h5', '_encoder.pkl')
            joblib.dump(self.label_encoder, encoder_path)
            
            logger.info(f"Model saved to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    async def generate_signal(self, market_data: pd.DataFrame) -> Optional[Signal]:
        """Generate trading signal from real-time market data"""
        try:
            if self.model is None or self.scaler is None:
                logger.warning("Model not loaded, cannot generate signals")
                return None
            
            # Engineer features
            df_features = self.engineer_features(market_data.copy())
            
            if len(df_features) < self.sequence_length:
                logger.warning("Insufficient data for signal generation")
                return None
            
            # Prepare latest sequence
            latest_sequence = df_features[self.feature_columns].iloc[-self.sequence_length:].values
            
            # Scale features
            latest_sequence_scaled = self.scaler.transform(latest_sequence.reshape(-1, latest_sequence.shape[-1]))
            latest_sequence_scaled = latest_sequence_scaled.reshape(1, self.sequence_length, -1)
            
            # Generate prediction
            prediction = self.model.predict(latest_sequence_scaled, verbose=0)
            prediction_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            # Map prediction to action
            action_map = {0: 'buy', 1: 'sell', 2: 'hold'}
            action = action_map[prediction_class]
            
            # Create signal
            signal = Signal(
                symbol=market_data['symbol'].iloc[-1] if 'symbol' in market_data.columns else 'EUR/USD',
                timestamp=datetime.utcnow(),
                action=action,
                confidence=confidence,
                price=market_data['close'].iloc[-1],
                features=df_features.iloc[-1].to_dict(),
                model_version=os.path.basename(self.model_path)
            )
            
            # Apply signal filtering
            filtered_signal = self._filter_signal(signal)
            
            if filtered_signal:
                self.last_signal = filtered_signal
                self.live_performance.append({
                    'timestamp': filtered_signal.timestamp,
                    'action': filtered_signal.action,
                    'confidence': filtered_signal.confidence,
                    'price': filtered_signal.price
                })
            
            return filtered_signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None
    
    def _filter_signal(self, signal: Signal) -> Optional[Signal]:
        """Filter signals to avoid noise"""
        try:
            # Add to history
            self.signal_history.append(signal)
            
            # Keep only recent signals
            if len(self.signal_history) > 10:
                self.signal_history = self.signal_history[-10:]
            
            # Check for consecutive signals
            if len(self.signal_history) >= self.min_consecutive_signals:
                recent_signals = self.signal_history[-self.min_consecutive_signals:]
                
                # Check if all recent signals are the same
                if all(s.action == recent_signals[0].action for s in recent_signals):
                    # Check confidence threshold
                    if signal.confidence > 0.6:  # 60% confidence threshold
                        return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Error filtering signal: {e}")
            return None
    
    async def start(self):
        """Start the AI signal generator"""
        self.is_running = True
        logger.info("AI Signal Generator started")
    
    async def stop(self):
        """Stop the AI signal generator"""
        self.is_running = False
        logger.info("AI Signal Generator stopped")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'backtest_results': self.backtest_results.__dict__ if self.backtest_results else None,
            'live_signals_count': len(self.live_performance),
            'last_signal': self.last_signal.__dict__ if self.last_signal else None,
            'model_path': self.model_path,
            'is_running': self.is_running
        }
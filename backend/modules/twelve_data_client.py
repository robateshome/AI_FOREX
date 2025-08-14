#!/usr/bin/env python3
"""
Twelve Data API Client Module
Provides historical and real-time Forex data via REST and WebSocket
"""

import asyncio
import aiohttp
import websockets
import json
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

@dataclass
class TwelveDataConfig:
    """Configuration for Twelve Data API"""
    api_key: str
    base_url: str = "https://api.twelvedata.com"
    websocket_url: str = "wss://ws.twelvedata.com/v1/quotes/price"
    rate_limit_per_minute: int = 800
    rate_limit_per_second: int = 8

class TwelveDataClient:
    """Client for Twelve Data API with rate limiting and error handling"""
    
    def __init__(self, config: TwelveDataConfig):
        self.config = config
        self.session = None
        self.websocket = None
        self.rate_limit_tracker = {
            'requests_this_minute': 0,
            'requests_this_second': 0,
            'last_minute_reset': time.time(),
            'last_second_reset': time.time()
        }
        self.is_connected = False
        self.data_callbacks: List[Callable] = []
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        if self.websocket:
            await self.websocket.close()
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        current_time = time.time()
        
        # Reset minute counter
        if current_time - self.rate_limit_tracker['last_minute_reset'] >= 60:
            self.rate_limit_tracker['requests_this_minute'] = 0
            self.rate_limit_tracker['last_minute_reset'] = current_time
        
        # Reset second counter
        if current_time - self.rate_limit_tracker['last_second_reset'] >= 1:
            self.rate_limit_tracker['requests_this_second'] = 0
            self.rate_limit_tracker['last_second_reset'] = current_time
        
        # Check limits
        if (self.rate_limit_tracker['requests_this_minute'] >= self.config.rate_limit_per_minute or
            self.rate_limit_tracker['requests_this_second'] >= self.config.rate_limit_per_second):
            return False
        
        return True
    
    def _increment_rate_limit(self):
        """Increment rate limit counters"""
        self.rate_limit_tracker['requests_this_minute'] += 1
        self.rate_limit_tracker['requests_this_second'] += 1
    
    async def get_historical_data(
        self, 
        symbol: str, 
        interval: str = "1min", 
        outputsize: int = 5000,
        timezone: str = "UTC"
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        try:
            if not self._check_rate_limit():
                logger.warning("Rate limit exceeded, waiting...")
                await asyncio.sleep(1)
                return await self.get_historical_data(symbol, interval, outputsize, timezone)
            
            url = f"{self.config.base_url}/time_series"
            params = {
                'symbol': symbol,
                'interval': interval,
                'outputsize': outputsize,
                'timezone': timezone,
                'apikey': self.config.api_key
            }
            
            async with self.session.get(url, params=params) as response:
                self._increment_rate_limit()
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('status') == 'ok':
                        # Convert to DataFrame
                        df = pd.DataFrame(data['values'])
                        
                        # Rename columns to standard format
                        column_mapping = {
                            'datetime': 'timestamp',
                            'open': 'open',
                            'high': 'high',
                            'low': 'low',
                            'close': 'close',
                            'volume': 'volume'
                        }
                        
                        df = df.rename(columns=column_mapping)
                        
                        # Convert data types
                        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                        for col in numeric_columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        # Convert timestamp
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        
                        # Add symbol column
                        df['symbol'] = symbol
                        
                        # Sort by timestamp
                        df = df.sort_values('timestamp').reset_index(drop=True)
                        
                        # Drop NaN values
                        df = df.dropna()
                        
                        logger.info(f"Fetched {len(df)} historical records for {symbol}")
                        return df
                    
                    else:
                        error_msg = data.get('message', 'Unknown error')
                        logger.error(f"API error: {error_msg}")
                        raise Exception(f"Twelve Data API error: {error_msg}")
                
                else:
                    logger.error(f"HTTP error {response.status}: {await response.text()}")
                    raise Exception(f"HTTP error {response.status}")
                    
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise
    
    async def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for a symbol"""
        try:
            if not self._check_rate_limit():
                logger.warning("Rate limit exceeded, waiting...")
                await asyncio.sleep(1)
                return await self.get_real_time_quote(symbol)
            
            url = f"{self.config.base_url}/quote"
            params = {
                'symbol': symbol,
                'apikey': self.config.api_key
            }
            
            async with self.session.get(url, params=params) as response:
                self._increment_rate_limit()
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('status') == 'ok':
                        return data
                    else:
                        error_msg = data.get('message', 'Unknown error')
                        logger.error(f"API error: {error_msg}")
                        raise Exception(f"Twelve Data API error: {error_msg}")
                
                else:
                    logger.error(f"HTTP error {response.status}: {await response.text()}")
                    raise Exception(f"HTTP error {response.status}")
                    
        except Exception as e:
            logger.error(f"Error fetching real-time quote for {symbol}: {e}")
            raise
    
    async def connect_websocket(self, symbols: List[str]):
        """Connect to WebSocket for real-time data"""
        try:
            # Create subscription message
            subscription = {
                "action": "subscribe",
                "params": {
                    "symbols": symbols
                }
            }
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(self.config.websocket_url)
            self.is_connected = True
            
            # Send subscription
            await self.websocket.send(json.dumps(subscription))
            logger.info(f"WebSocket connected and subscribed to {symbols}")
            
            # Start listening for messages
            asyncio.create_task(self._websocket_listener())
            
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {e}")
            self.is_connected = False
            raise
    
    async def _websocket_listener(self):
        """Listen for WebSocket messages"""
        try:
            while self.is_connected and self.websocket:
                try:
                    message = await self.websocket.recv()
                    data = json.loads(message)
                    
                    # Process the message
                    await self._process_websocket_message(data)
                    
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed")
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from WebSocket: {e}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    
        except Exception as e:
            logger.error(f"WebSocket listener error: {e}")
        finally:
            self.is_connected = False
    
    async def _process_websocket_message(self, data: Dict[str, Any]):
        """Process incoming WebSocket message"""
        try:
            if 'data' in data:
                # Extract quote data
                quote = data['data']
                
                # Convert to standard format
                market_data = {
                    'symbol': quote.get('symbol'),
                    'timestamp': datetime.fromtimestamp(int(quote.get('timestamp', 0)) / 1000),
                    'price': float(quote.get('price', 0)),
                    'bid': float(quote.get('bid', 0)),
                    'ask': float(quote.get('ask', 0)),
                    'volume': float(quote.get('volume', 0))
                }
                
                # Notify callbacks
                for callback in self.data_callbacks:
                    try:
                        await callback(market_data)
                    except Exception as e:
                        logger.error(f"Error in data callback: {e}")
                
                logger.debug(f"Processed real-time data for {market_data['symbol']}")
                
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def add_data_callback(self, callback: Callable):
        """Add callback for real-time data"""
        self.data_callbacks.append(callback)
    
    def remove_data_callback(self, callback: Callable):
        """Remove callback for real-time data"""
        if callback in self.data_callbacks:
            self.data_callbacks.remove(callback)
    
    async def disconnect_websocket(self):
        """Disconnect from WebSocket"""
        try:
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
                self.is_connected = False
                logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket: {e}")
    
    async def get_multiple_symbols_data(
        self, 
        symbols: List[str], 
        interval: str = "1min", 
        outputsize: int = 1000
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols concurrently"""
        try:
            tasks = []
            for symbol in symbols:
                task = self.get_historical_data(symbol, interval, outputsize)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            data_dict = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching data for {symbols[i]}: {result}")
                else:
                    data_dict[symbols[i]] = result
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Error fetching multiple symbols data: {e}")
            raise
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status"""
        return {
            'requests_this_minute': self.rate_limit_tracker['requests_this_minute'],
            'requests_this_second': self.rate_limit_tracker['requests_this_second'],
            'minute_limit': self.config.rate_limit_per_minute,
            'second_limit': self.config.rate_limit_per_second,
            'is_connected': self.is_connected
        }

# Factory function to create client from environment
def create_twelve_data_client() -> TwelveDataClient:
    """Create Twelve Data client from environment variables"""
    load_dotenv()
    
    api_key = os.getenv('TWELVE_DATA_API_KEY')
    if not api_key:
        raise ValueError("TWELVE_DATA_API_KEY environment variable not set")
    
    config = TwelveDataConfig(api_key=api_key)
    return TwelveDataClient(config)
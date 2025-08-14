#!/usr/bin/env python3
"""
CLI Dashboard for AI Forex Signal Generator
Real-time monitoring and control interface
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv
import curses
import threading
import queue

class CLIDashboard:
    """Command-line dashboard for monitoring the AI Forex signal generator"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.session = None
        self.is_running = False
        self.update_interval = 2.0  # Update every 2 seconds
        
        # Data storage
        self.current_status = {}
        self.recent_signals = []
        self.performance_metrics = {}
        self.market_data = {}
        
        # Update queue for thread safety
        self.update_queue = queue.Queue()
        
        # Load environment
        load_dotenv()
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def start(self):
        """Start the CLI dashboard"""
        try:
            self.is_running = True
            logger.info("Starting CLI Dashboard...")
            
            # Test API connection
            if not await self._test_connection():
                logger.error("Failed to connect to API")
                return
            
            # Start data update loop
            asyncio.create_task(self._data_update_loop())
            
            # Start the dashboard
            await self._run_dashboard()
            
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
            raise
    
    async def stop(self):
        """Stop the CLI dashboard"""
        self.is_running = False
        logger.info("CLI Dashboard stopped")
    
    async def _test_connection(self) -> bool:
        """Test connection to the API"""
        try:
            async with self.session.get(f"{self.api_base_url}/health") as response:
                if response.status == 200:
                    logger.info("API connection successful")
                    return True
                else:
                    logger.error(f"API health check failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False
    
    async def _data_update_loop(self):
        """Continuous loop for updating data"""
        while self.is_running:
            try:
                await self._fetch_all_data()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in data update loop: {e}")
                await asyncio.sleep(5)
    
    async def _fetch_all_data(self):
        """Fetch all data from the API"""
        try:
            # Fetch status
            async with self.session.get(f"{self.api_base_url}/api/status") as response:
                if response.status == 200:
                    self.current_status = await response.json()
            
            # Fetch AI signals
            async with self.session.get(f"{self.api_base_url}/api/ai/signals") as response:
                if response.status == 200:
                    data = await response.json()
                    self.recent_signals = data.get('signals', [])
            
            # Fetch performance metrics
            async with self.session.get(f"{self.api_base_url}/api/ai/performance") as response:
                if response.status == 200:
                    self.performance_metrics = await response.json()
            
            # Fetch market data
            async with self.session.get(f"{self.api_base_url}/api/ai/market-data") as response:
                if response.status == 200:
                    data = await response.json()
                    self.market_data = data.get('market_data', {})
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
    
    async def _run_dashboard(self):
        """Run the main dashboard interface"""
        try:
            # Use curses for the dashboard
            curses.wrapper(self._dashboard_main)
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            # Fallback to simple console output
            await self._simple_dashboard()
    
    def _dashboard_main(self, stdscr):
        """Main dashboard function using curses"""
        # Setup curses
        curses.curs_set(0)  # Hide cursor
        stdscr.clear()
        stdscr.refresh()
        
        # Color setup
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)   # Success
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)     # Error
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Warning
        curses.init_pair(4, curses.COLOR_BLUE, curses.COLOR_BLACK)    # Info
        curses.init_pair(5, curses.COLOR_CYAN, curses.COLOR_BLACK)    # Highlight
        
        try:
            while self.is_running:
                self._draw_dashboard(stdscr)
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            curses.endwin()
    
    def _draw_dashboard(self, stdscr):
        """Draw the dashboard content"""
        try:
            stdscr.clear()
            height, width = stdscr.getmaxyx()
            
            # Title
            title = "🤖 AI Forex Signal Generator Dashboard"
            stdscr.addstr(0, (width - len(title)) // 2, title, curses.color_pair(5) | curses.A_BOLD)
            
            # Status section
            self._draw_status_section(stdscr, 2, width)
            
            # Performance section
            self._draw_performance_section(stdscr, 8, width)
            
            # Recent signals section
            self._draw_signals_section(stdscr, 16, width)
            
            # Market data section
            self._draw_market_section(stdscr, 24, width)
            
            # Footer
            footer = f"Press Ctrl+C to exit | Last update: {datetime.now().strftime('%H:%M:%S')}"
            stdscr.addstr(height - 1, 0, footer, curses.color_pair(4))
            
            stdscr.refresh()
            
        except Exception as e:
            stdscr.addstr(0, 0, f"Dashboard error: {e}", curses.color_pair(2))
            stdscr.refresh()
    
    def _draw_status_section(self, stdscr, y: int, width: int):
        """Draw the status section"""
        try:
            stdscr.addstr(y, 0, "📊 SYSTEM STATUS", curses.color_pair(5) | curses.A_BOLD)
            
            if self.current_status:
                status = self.current_status.get('system_status', 'unknown')
                color = curses.color_pair(1) if status == 'operational' else curses.color_pair(2)
                stdscr.addstr(y + 1, 2, f"Status: {status.upper()}", color)
                
                # AI Pipeline status
                ai_status = self.current_status.get('ai_pipeline', {})
                if ai_status:
                    uptime = ai_status.get('uptime_seconds', 0)
                    uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"
                    stdscr.addstr(y + 2, 2, f"Uptime: {uptime_str}")
                    
                    total_signals = ai_status.get('total_signals', 0)
                    stdscr.addstr(y + 3, 2, f"Total Signals: {total_signals}")
                    
                    avg_latency = ai_status.get('avg_latency_ms', 0)
                    stdscr.addstr(y + 4, 2, f"Avg Latency: {avg_latency:.1f}ms")
            else:
                stdscr.addstr(y + 1, 2, "Status: Loading...", curses.color_pair(3))
                
        except Exception as e:
            stdscr.addstr(y + 1, 2, f"Status error: {e}", curses.color_pair(2))
    
    def _draw_performance_section(self, stdscr, y: int, width: int):
        """Draw the performance section"""
        try:
            stdscr.addstr(y, 0, "🎯 PERFORMANCE METRICS", curses.color_pair(5) | curses.A_BOLD)
            
            if self.performance_metrics:
                model_perf = self.performance_metrics.get('model_performance', {})
                if model_perf:
                    backtest = model_perf.get('backtest_results', {})
                    if backtest:
                        accuracy = backtest.get('accuracy', 0)
                        stdscr.addstr(y + 1, 2, f"Accuracy: {accuracy:.2%}")
                        
                        sharpe = backtest.get('sharpe_ratio', 0)
                        stdscr.addstr(y + 2, 2, f"Sharpe Ratio: {sharpe:.3f}")
                        
                        win_rate = backtest.get('win_rate', 0)
                        stdscr.addstr(y + 3, 2, f"Win Rate: {win_rate:.2%}")
                        
                        max_dd = backtest.get('max_drawdown', 0)
                        stdscr.addstr(y + 4, 2, f"Max Drawdown: {max_dd:.2%}")
                else:
                    stdscr.addstr(y + 1, 2, "Performance: No data available", curses.color_pair(3))
            else:
                stdscr.addstr(y + 1, 2, "Performance: Loading...", curses.color_pair(3))
                
        except Exception as e:
            stdscr.addstr(y + 1, 2, f"Performance error: {e}", curses.color_pair(2))
    
    def _draw_signals_section(self, stdscr, y: int, width: int):
        """Draw the recent signals section"""
        try:
            stdscr.addstr(y, 0, "📈 RECENT SIGNALS", curses.color_pair(5) | curses.A_BOLD)
            
            if self.recent_signals:
                for i, signal in enumerate(self.recent_signals[:5]):  # Show last 5 signals
                    if y + i + 1 < curses.LINES - 2:  # Check bounds
                        symbol = signal.get('symbol', 'UNKNOWN')
                        action = signal.get('action', 'UNKNOWN').upper()
                        confidence = signal.get('confidence', 0)
                        price = signal.get('price', 0)
                        timestamp = signal.get('timestamp', '')
                        
                        # Format timestamp
                        try:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            time_str = dt.strftime('%H:%M:%S')
                        except:
                            time_str = timestamp
                        
                        # Color based on action
                        action_color = curses.color_pair(1) if action == 'BUY' else curses.color_pair(2) if action == 'SELL' else curses.color_pair(3)
                        
                        signal_str = f"{time_str} {symbol} {action} {price:.5f} ({confidence:.1%})"
                        stdscr.addstr(y + i + 1, 2, signal_str, action_color)
            else:
                stdscr.addstr(y + 1, 2, "Signals: No recent signals", curses.color_pair(3))
                
        except Exception as e:
            stdscr.addstr(y + 1, 2, f"Signals error: {e}", curses.color_pair(2))
    
    def _draw_market_section(self, stdscr, y: int, width: int):
        """Draw the market data section"""
        try:
            stdscr.addstr(y, 0, "💱 MARKET DATA", curses.color_pair(5) | curses.A_BOLD)
            
            if self.market_data:
                for i, (symbol, data) in enumerate(list(self.market_data.items())[:3]):  # Show first 3 symbols
                    if y + i + 1 < curses.LINES - 2:  # Check bounds
                        last_price = data.get('last_price', 0)
                        price_change_1h = data.get('price_change_1h', 0)
                        price_change_24h = data.get('price_change_24h', 0)
                        
                        # Color based on price change
                        change_color = curses.color_pair(1) if price_change_1h > 0 else curses.color_pair(2) if price_change_1h < 0 else curses.color_pair(3)
                        
                        market_str = f"{symbol}: {last_price:.5f} (1h: {price_change_1h:+.3f}%, 24h: {price_change_24h:+.3f}%)"
                        stdscr.addstr(y + i + 1, 2, market_str, change_color)
            else:
                stdscr.addstr(y + 1, 2, "Market: No data available", curses.color_pair(3))
                
        except Exception as e:
            stdscr.addstr(y + 1, 2, f"Market error: {e}", curses.color_pair(2))
    
    async def _simple_dashboard(self):
        """Simple console-based dashboard fallback"""
        try:
            while self.is_running:
                os.system('clear' if os.name == 'posix' else 'cls')
                
                print("🤖 AI Forex Signal Generator Dashboard")
                print("=" * 50)
                
                # Status
                if self.current_status:
                    status = self.current_status.get('system_status', 'unknown')
                    print(f"📊 Status: {status.upper()}")
                    
                    ai_status = self.current_status.get('ai_pipeline', {})
                    if ai_status:
                        uptime = ai_status.get('uptime_seconds', 0)
                        uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"
                        print(f"⏱️  Uptime: {uptime_str}")
                        print(f"📈 Total Signals: {ai_status.get('total_signals', 0)}")
                        print(f"⚡ Avg Latency: {ai_status.get('avg_latency_ms', 0):.1f}ms")
                
                # Performance
                if self.performance_metrics:
                    model_perf = self.performance_metrics.get('model_performance', {})
                    if model_perf:
                        backtest = model_perf.get('backtest_results', {})
                        if backtest:
                            print(f"\n🎯 Performance:")
                            print(f"   Accuracy: {backtest.get('accuracy', 0):.2%}")
                            print(f"   Sharpe Ratio: {backtest.get('sharpe_ratio', 0):.3f}")
                            print(f"   Win Rate: {backtest.get('win_rate', 0):.2%}")
                
                # Recent signals
                if self.recent_signals:
                    print(f"\n📈 Recent Signals:")
                    for signal in self.recent_signals[:5]:
                        symbol = signal.get('symbol', 'UNKNOWN')
                        action = signal.get('action', 'UNKNOWN').upper()
                        confidence = signal.get('confidence', 0)
                        price = signal.get('price', 0)
                        timestamp = signal.get('timestamp', '')
                        
                        try:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            time_str = dt.strftime('%H:%M:%S')
                        except:
                            time_str = timestamp
                        
                        print(f"   {time_str} {symbol} {action} {price:.5f} ({confidence:.1%})")
                
                # Market data
                if self.market_data:
                    print(f"\n💱 Market Data:")
                    for symbol, data in list(self.market_data.items())[:3]:
                        last_price = data.get('last_price', 0)
                        price_change_1h = data.get('price_change_1h', 0)
                        price_change_24h = data.get('price_change_24h', 0)
                        print(f"   {symbol}: {last_price:.5f} (1h: {price_change_1h:+.3f}%, 24h: {price_change_24h:+.3f}%)")
                
                print(f"\n⏰ Last update: {datetime.now().strftime('%H:%M:%S')}")
                print("Press Ctrl+C to exit")
                
                await asyncio.sleep(2)
                
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped")
        except Exception as e:
            print(f"❌ Dashboard error: {e}")

async def main():
    """Main function"""
    dashboard = CLIDashboard()
    async with dashboard:
        await dashboard.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
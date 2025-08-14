#!/usr/bin/env python3
"""
AI Forex Signal Generator - Demo Mode
Works without requiring API calls for demonstration purposes
"""

import asyncio
import logging
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
import time
import random
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global demo data
demo_pipeline_metrics = {
    "total_signals": 0,
    "avg_latency_ms": 0,
    "signals_per_hour": 0,
    "last_signal_time": None,
    "uptime_seconds": 0,
    "data_feed_status": "demo_mode",
    "model_status": "running"
}

demo_signals = []
demo_market_data = {
    "EUR/USD": {
        "last_price": 1.16468,
        "price_change_1h": 0.023,
        "price_change_24h": -0.156,
        "data_points": 1500,
        "last_update": datetime.now()
    }
}

demo_model_performance = {
    "backtest_results": {
        "accuracy": 0.78,
        "precision": 0.82,
        "recall": 0.75,
        "f1_score": 0.78,
        "sharpe_ratio": 1.45,
        "max_drawdown": 0.08,
        "win_rate": 0.68,
        "total_trades": 156,
        "profitable_trades": 106
    }
}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
                await self.disconnect(connection)

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting AI Forex Signal Generator in DEMO MODE...")
    
    # Start demo data generation
    asyncio.create_task(demo_data_generator())
    
    logger.info("AI Signal Pipeline started successfully in DEMO MODE")
    
    yield
    
    # Cleanup
    logger.info("Shutting down AI Forex Signal Generator...")

async def demo_data_generator():
    """Generate demo data for the dashboard"""
    global demo_pipeline_metrics, demo_signals, demo_market_data
    
    start_time = time.time()
    
    while True:
        try:
            # Update uptime
            demo_pipeline_metrics["uptime_seconds"] = time.time() - start_time
            
            # Generate demo signals every 30 seconds
            if len(demo_signals) < 10 or random.random() < 0.1:
                signal = generate_demo_signal()
                demo_signals.insert(0, signal)
                demo_signals = demo_signals[:20]  # Keep last 20 signals
                
                demo_pipeline_metrics["total_signals"] = len(demo_signals)
                demo_pipeline_metrics["last_signal_time"] = signal["timestamp"]
                demo_pipeline_metrics["signals_per_hour"] = len(demo_signals) * 2  # Estimate
            
            # Update market data
            for symbol in demo_market_data:
                price_change = random.uniform(-0.001, 0.001)
                demo_market_data[symbol]["last_price"] += price_change
                demo_market_data[symbol]["last_update"] = datetime.now()
            
            # Update latency (simulate real system)
            demo_pipeline_metrics["avg_latency_ms"] = random.uniform(45, 120)
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            logger.error(f"Error in demo data generator: {e}")
            await asyncio.sleep(5)

def generate_demo_signal():
    """Generate a realistic demo trading signal"""
    actions = ["buy", "sell", "hold"]
    weights = [0.4, 0.4, 0.2]  # 40% buy, 40% sell, 20% hold
    
    action = random.choices(actions, weights=weights)[0]
    confidence = random.uniform(0.65, 0.95)
    
    # Generate realistic price based on action
    base_price = 1.16468
    if action == "buy":
        price = base_price + random.uniform(0.0001, 0.001)
    elif action == "sell":
        price = base_price - random.uniform(0.0001, 0.001)
    else:
        price = base_price + random.uniform(-0.0005, 0.0005)
    
    return {
        "id": f"demo_{int(time.time())}",
        "symbol": "EUR/USD",
        "action": action,
        "price": round(price, 5),
        "confidence": round(confidence, 3),
        "timestamp": datetime.now().isoformat(),
        "source": "demo_ai_model"
    }

# Create FastAPI app
app = FastAPI(
    title="AI Forex Signal Generator - Demo Mode",
    description="Real-time AI-powered Forex trading signal generator (Demo Mode)",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "AI Forex Signal Generator API - Demo Mode", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ai_pipeline": True,
        "demo_mode": True,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/status")
async def get_status():
    """Get detailed system status"""
    return {
        "system_status": "operational",
        "demo_mode": True,
        "ai_pipeline": demo_pipeline_metrics,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/ai/signals")
async def get_ai_signals(limit: int = 10):
    """Get recent AI-generated trading signals"""
    return {
        "signals": demo_signals[:limit],
        "total_count": len(demo_signals),
        "demo_mode": True,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/ai/performance")
async def get_ai_performance():
    """Get AI model performance metrics"""
    return {
        "pipeline_metrics": demo_pipeline_metrics,
        "model_performance": demo_model_performance,
        "market_summary": demo_market_data,
        "demo_mode": True,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/ai/retrain")
async def retrain_ai_model():
    """Simulate model retraining"""
    return {
        "status": "success", 
        "message": "Demo model retraining completed (simulated)",
        "demo_mode": True
    }

@app.get("/api/ai/market-data")
async def get_market_data_summary():
    """Get current market data summary"""
    return {
        "market_data": demo_market_data,
        "demo_mode": True,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(
        "main_ai_demo:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
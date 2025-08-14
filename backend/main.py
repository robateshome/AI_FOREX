#!/usr/bin/env python3
"""
Forex Trading Bot - Main Application
Module 0x01: Main Application Entry Point
CRC32: PLACEHOLDER_CRC32_MAIN
"""

import asyncio
import logging
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from modules.data_feed import DataFeedManager
from modules.divergence_detector import DivergenceDetector
from modules.signal_engine import SignalEngine
from modules.integrity_validation import IntegrityValidator
from modules.failsafe_execution import FailsafeManager
from modules.signal_pipeline import RealTimeSignalPipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global managers
data_feed_manager = None
divergence_detector = None
signal_engine = None
integrity_validator = None
failsafe_manager = None
ai_signal_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global data_feed_manager, divergence_detector, signal_engine, integrity_validator, failsafe_manager, ai_signal_pipeline
    
    logger.info("Starting AI Forex Signal Generator...")
    
    # Initialize all modules
    try:
        # Initialize integrity validator first
        integrity_validator = IntegrityValidator()
        
        # Initialize data feed manager
        data_feed_manager = DataFeedManager()
        
        # Initialize divergence detector
        divergence_detector = DivergenceDetector()
        
        # Initialize signal engine
        signal_engine = SignalEngine()
        
        # Initialize failsafe manager
        failsafe_manager = FailsafeManager()
        
        # Initialize AI signal pipeline
        ai_signal_pipeline = RealTimeSignalPipeline(symbols=["EUR/USD"], update_interval=1.0)
        await ai_signal_pipeline.initialize()
        
        # Start all services
        await asyncio.gather(
            data_feed_manager.start(),
            divergence_detector.start(),
            signal_engine.start(),
            failsafe_manager.start(),
            ai_signal_pipeline.start()
        )
        
        logger.info("All modules started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start modules: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down AI Forex Signal Generator...")
    try:
        await asyncio.gather(
            data_feed_manager.stop(),
            divergence_detector.stop(),
            signal_engine.stop(),
            failsafe_manager.stop(),
            ai_signal_pipeline.stop()
        )
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Create FastAPI app
app = FastAPI(
    title="Forex Trading Bot",
    description="Real-time Forex trading bot with divergence detection",
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

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Forex Trading Bot API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "modules": {
            "data_feed": data_feed_manager.is_running() if data_feed_manager else False,
            "divergence_detector": divergence_detector.is_running() if divergence_detector else False,
            "signal_engine": signal_engine.is_running() if signal_engine else False,
            "integrity_validator": integrity_validator.is_valid() if integrity_validator else False,
            "failsafe_manager": failsafe_manager.is_running() if failsafe_manager else False
        }
    }

@app.get("/api/status")
async def get_status():
    """Get detailed system status"""
    return {
        "system_status": "operational",
        "data_sources": data_feed_manager.get_status() if data_feed_manager else {},
        "signals": signal_engine.get_recent_signals() if signal_engine else [],
        "divergences": divergence_detector.get_recent_divergences() if divergence_detector else [],
        "ai_pipeline": ai_signal_pipeline.get_pipeline_metrics().__dict__ if ai_signal_pipeline else {}
    }

@app.get("/api/ai/signals")
async def get_ai_signals(limit: int = 10):
    """Get recent AI-generated trading signals"""
    if not ai_signal_pipeline:
        return {"error": "AI pipeline not initialized"}
    
    signals = ai_signal_pipeline.get_recent_signals(limit=limit)
    return {
        "signals": signals,
        "total_count": len(signals),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/ai/performance")
async def get_ai_performance():
    """Get AI model performance metrics"""
    if not ai_signal_pipeline:
        return {"error": "AI pipeline not initialized"}
    
    metrics = ai_signal_pipeline.get_pipeline_metrics()
    return {
        "pipeline_metrics": metrics.__dict__,
        "model_performance": ai_signal_pipeline.ai_generator.get_performance_summary() if ai_signal_pipeline.ai_generator else {},
        "market_summary": ai_signal_pipeline.get_market_data_summary()
    }

@app.post("/api/ai/retrain")
async def retrain_ai_model():
    """Retrain the AI model with latest data"""
    if not ai_signal_pipeline:
        return {"error": "AI pipeline not initialized"}
    
    try:
        await ai_signal_pipeline.retrain_model()
        return {"status": "success", "message": "Model retraining completed"}
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/ai/market-data")
async def get_market_data_summary():
    """Get current market data summary"""
    if not ai_signal_pipeline:
        return {"error": "AI pipeline not initialized"}
    
    summary = ai_signal_pipeline.get_market_data_summary()
    return {
        "market_data": summary,
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

@app.post("/api/configure")
async def configure_bot(config: dict):
    """Configure bot parameters"""
    try:
        # Update configuration
        if data_feed_manager:
            await data_feed_manager.update_config(config.get("data_feed", {}))
        if divergence_detector:
            await divergence_detector.update_config(config.get("divergence", {}))
        
        return {"status": "success", "message": "Configuration updated"}
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
#!/usr/bin/env python3
"""
AI Forex Signal Generator - Simplified Main Application
Focuses on AI signal generation without legacy modules
"""

import asyncio
import logging
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

# Import only the AI modules
from modules.ai_signal_generator import AISignalGenerator
from modules.twelve_data_client import create_twelve_data_client
from modules.telegram_alerts import create_telegram_alerts
from modules.signal_pipeline import RealTimeSignalPipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global AI pipeline
ai_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global ai_pipeline
    
    logger.info("Starting AI Forex Signal Generator...")
    
    try:
        # Initialize AI signal pipeline
        ai_pipeline = RealTimeSignalPipeline(symbols=["EUR/USD"], update_interval=1.0)
        await ai_pipeline.initialize()
        await ai_pipeline.start()
        
        logger.info("AI Signal Pipeline started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start AI pipeline: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down AI Forex Signal Generator...")
    try:
        if ai_pipeline:
            await ai_pipeline.stop()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Create FastAPI app
app = FastAPI(
    title="AI Forex Signal Generator",
    description="Real-time AI-powered Forex trading signal generator",
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
    return {"message": "AI Forex Signal Generator API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ai_pipeline": ai_pipeline.is_running if ai_pipeline else False,
        "timestamp": "2025-01-27T20:00:00Z"
    }

@app.get("/api/status")
async def get_status():
    """Get detailed system status"""
    if not ai_pipeline:
        return {"error": "AI pipeline not initialized"}
    
    return {
        "system_status": "operational",
        "ai_pipeline": ai_pipeline.get_pipeline_metrics().__dict__,
        "timestamp": "2025-01-27T20:00:00Z"
    }

@app.get("/api/ai/signals")
async def get_ai_signals(limit: int = 10):
    """Get recent AI-generated trading signals"""
    if not ai_pipeline:
        return {"error": "AI pipeline not initialized"}
    
    signals = ai_pipeline.get_recent_signals(limit=limit)
    return {
        "signals": signals,
        "total_count": len(signals),
        "timestamp": "2025-01-27T20:00:00Z"
    }

@app.get("/api/ai/performance")
async def get_ai_performance():
    """Get AI model performance metrics"""
    if not ai_pipeline:
        return {"error": "AI pipeline not initialized"}
    
    metrics = ai_pipeline.get_pipeline_metrics()
    return {
        "pipeline_metrics": metrics.__dict__,
        "model_performance": ai_pipeline.ai_generator.get_performance_summary() if ai_pipeline.ai_generator else {},
        "market_summary": ai_pipeline.get_market_data_summary()
    }

@app.post("/api/ai/retrain")
async def retrain_ai_model():
    """Retrain the AI model with latest data"""
    if not ai_pipeline:
        return {"error": "AI pipeline not initialized"}
    
    try:
        await ai_pipeline.retrain_model()
        return {"status": "success", "message": "Model retraining completed"}
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/ai/market-data")
async def get_market_data_summary():
    """Get current market data summary"""
    if not ai_pipeline:
        return {"error": "AI pipeline not initialized"}
    
    summary = ai_pipeline.get_market_data_summary()
    return {
        "market_data": summary,
        "timestamp": "2025-01-27T20:00:00Z"
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
        "main_ai:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
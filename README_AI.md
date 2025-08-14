# 🤖 AI Forex Signal Generator

A real-time AI-powered Forex trading signal generator that uses LSTM neural networks to analyze market data and generate buy/sell/hold signals with sub-2-second latency.

## 🚀 Features

- **Real-time Signal Generation**: LSTM-based AI model generates trading signals in <2 seconds
- **Live Market Data**: Integrates with Twelve Data API for real-time Forex data
- **Technical Analysis**: Comprehensive feature engineering with SMA, EMA, RSI, MACD, ATR
- **Signal Filtering**: Requires consecutive predictions to avoid noise
- **Performance Metrics**: Backtesting with Sharpe ratio, drawdown, win rate
- **Telegram Alerts**: Real-time signal notifications via Telegram bot
- **CLI Dashboard**: Real-time monitoring interface
- **Cloud Ready**: Docker containerization for easy deployment
- **Auto-retraining**: Scheduled model retraining with latest data

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Twelve Data  │    │   AI Signal      │    │   Telegram      │
│     API        │───▶│   Generator      │───▶│     Alerts      │
│  (Real-time)   │    │   (LSTM)         │    │   (Optional)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Market Data  │    │   Signal         │    │   Performance   │
│     Cache      │    │   Pipeline       │    │   Metrics       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🛠️ Technology Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn
- **AI/ML**: TensorFlow 2.15, Keras, Scikit-learn
- **Data Processing**: Pandas, NumPy, TA-Lib
- **Real-time**: WebSockets, AsyncIO
- **Data Source**: Twelve Data API
- **Notifications**: Telegram Bot API
- **Deployment**: Docker, Docker Compose
- **Monitoring**: CLI Dashboard, Health Checks

## 📋 Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Twelve Data API key
- Telegram Bot token (optional)
- 4GB+ RAM recommended
- Stable internet connection

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ai-forex-signals
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env
```

Required environment variables:
```bash
TWELVE_DATA_API_KEY=your_api_key_here
TELEGRAM_BOT_TOKEN=your_bot_token_here  # Optional
TELEGRAM_CHAT_ID=your_chat_id_here      # Optional
```

### 3. Docker Deployment (Recommended)

```bash
# Build and start services
docker-compose -f docker-compose.ai.yml up -d

# Check logs
docker-compose -f docker-compose.ai.yml logs -f ai-forex-signals

# Stop services
docker-compose -f docker-compose.ai.yml down
```

### 4. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# In another terminal, start the CLI dashboard
python cli_dashboard.py
```

## 📊 API Endpoints

### Core Endpoints

- `GET /` - API status
- `GET /health` - Health check
- `GET /api/status` - System status

### AI Signal Endpoints

- `GET /api/ai/signals?limit=10` - Recent trading signals
- `GET /api/ai/performance` - Model performance metrics
- `POST /api/ai/retrain` - Retrain AI model
- `GET /api/ai/market-data` - Current market data summary

### WebSocket

- `WS /ws` - Real-time data stream

## 🎯 Signal Generation Process

### 1. Data Acquisition
- Fetches 1-minute OHLCV data from Twelve Data API
- Maintains rolling cache of 2000 data points
- Real-time updates via REST API calls

### 2. Feature Engineering
- **Technical Indicators**: SMA(10), EMA(20), RSI(14), MACD, ATR(14)
- **Lag Features**: Last 15 close prices
- **Price Features**: Log returns, volume normalization
- **Data Preprocessing**: NaN removal, timestamp alignment, UTC conversion

### 3. AI Model
- **Architecture**: LSTM(128) → LSTM(64) → Dense(32) → Softmax(3)
- **Output**: Buy/Sell/Hold with confidence scores
- **Training**: 50 epochs with Adam optimizer
- **Validation**: 20% chronological split

### 4. Signal Filtering
- Minimum confidence threshold: 60%
- Consecutive signal requirement: 2 identical predictions
- Cooldown period: 60 seconds between signals per symbol

## 📈 Performance Metrics

- **Accuracy**: Model prediction accuracy
- **Precision/Recall**: Signal quality metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Maximum loss from peak
- **Win Rate**: Percentage of profitable signals
- **Latency**: Signal generation time

## 🔧 Configuration

### Model Parameters

```python
SEQUENCE_LENGTH = 60        # Minutes of historical data
FEATURE_WINDOW = 15         # Lag features window
MIN_CONFIDENCE = 0.6        # Minimum confidence threshold
MIN_CONSECUTIVE_SIGNALS = 2 # Required consecutive signals
```

### Trading Parameters

```python
SYMBOLS = ["EUR/USD"]       # Currency pairs to monitor
UPDATE_INTERVAL = 1.0       # Data update frequency (seconds)
SIGNAL_COOLDOWN = 60        # Cooldown between signals (seconds)
```

## 📱 Telegram Integration

### Setup

1. Create a bot with [@BotFather](https://t.me/botfather)
2. Get your bot token
3. Start a chat with your bot
4. Get your chat ID
5. Add to `.env` file

### Alert Types

- **Signal Alerts**: Real-time trading signals
- **Performance Updates**: Model performance metrics
- **Error Alerts**: System errors and warnings
- **Info Alerts**: System status updates

## 🖥️ CLI Dashboard

The CLI dashboard provides real-time monitoring:

```bash
python cli_dashboard.py
```

Features:
- System status and uptime
- Performance metrics
- Recent trading signals
- Live market data
- Color-coded information

## 🐳 Docker Deployment

### Production Deployment

```bash
# Build and deploy
docker-compose -f docker-compose.ai.yml up -d

# Scale if needed
docker-compose -f docker-compose.ai.yml up -d --scale ai-forex-signals=2

# Monitor logs
docker-compose -f docker-compose.ai.yml logs -f
```

### Health Checks

- API health endpoint: `/health`
- Docker health checks every 30 seconds
- Automatic restart on failure

## 🔒 Security

- Non-root Docker containers
- Environment variable configuration
- API rate limiting
- Input validation
- Secure API key storage

## 📊 Monitoring & Logging

### Logs

- Application logs: `logs/ai_forex_signals.log`
- Docker logs: `docker-compose logs ai-forex-signals`
- Log rotation: 100MB max, 5 backups

### Metrics

- Real-time performance tracking
- Signal generation latency
- API response times
- Error rates and types

## 🚨 Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Check Twelve Data API key
   - Verify internet connection
   - Check rate limits

2. **Model Training Failed**
   - Ensure sufficient historical data
   - Check memory availability
   - Verify data quality

3. **Telegram Alerts Not Working**
   - Verify bot token and chat ID
   - Check bot permissions
   - Test with simple message

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python -m uvicorn main:app --log-level debug
```

## 🔄 Model Retraining

### Automatic Retraining

- Triggered via API: `POST /api/ai/retrain`
- Uses latest market data
- Preserves model performance history
- Sends performance alerts

### Manual Retraining

```python
# Via Python
from modules.signal_pipeline import RealTimeSignalPipeline

pipeline = RealTimeSignalPipeline()
await pipeline.retrain_model()
```

## 📈 Backtesting

### Performance Metrics

- **Accuracy**: Overall prediction accuracy
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Maximum loss from peak
- **Win Rate**: Percentage of profitable trades
- **Total Trades**: Number of signals generated

### Backtest Results

Results are stored and can be accessed via:
- API endpoint: `/api/ai/performance`
- CLI dashboard
- Telegram alerts

## 🌐 Cloud Deployment

### AWS Deployment

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

# Deploy with ECS
aws ecs create-service --cluster forex-cluster --service-name ai-forex-signals --task-definition ai-forex-task
```

### GCP Deployment

```bash
# Build and push to GCR
docker tag ai-forex-signals gcr.io/<project>/ai-forex-signals
docker push gcr.io/<project>/ai-forex-signals

# Deploy to Cloud Run
gcloud run deploy ai-forex-signals --image gcr.io/<project>/ai-forex-signals
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not financial advice. Trading Forex involves substantial risk and may not be suitable for all investors. Past performance does not guarantee future results.

## 🆘 Support

- **Issues**: GitHub Issues
- **Documentation**: This README
- **Community**: GitHub Discussions

## 🔮 Roadmap

- [ ] Multi-currency support
- [ ] Advanced ML models (Transformers, CNNs)
- [ ] Reinforcement learning integration
- [ ] Walk-forward analysis
- [ ] Risk management system
- [ ] Paper trading mode
- [ ] Web-based dashboard
- [ ] Mobile app
- [ ] Advanced backtesting
- [ ] Portfolio optimization

---

**Built with ❤️ for the Forex trading community**
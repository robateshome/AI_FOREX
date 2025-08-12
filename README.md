# Forex Trading Bot with Divergence Detection

A comprehensive, real-time Forex trading bot system with advanced divergence detection capabilities, built using Python FastAPI backend and modern web frontend.

## 🚀 Features

### Core Trading Capabilities
- **Real-time Market Data**: Live OHLCV data from Twelve Data API with fallback mechanisms
- **Divergence Detection**: Advanced bullish/bearish divergence detection using RSI, MACD, Stochastic, and CCI indicators
- **Signal Generation**: Automated trading signal generation with confidence scoring
- **Multi-timeframe Analysis**: Support for 1m, 5m, 15m, 1h, and 4h timeframes

### Technical Indicators
- **RSI (Relative Strength Index)**: Momentum oscillator for overbought/oversold conditions
- **MACD (Moving Average Convergence Divergence)**: Trend-following momentum indicator
- **Stochastic Oscillator**: Momentum indicator comparing closing price to price range
- **CCI (Commodity Channel Index)**: Cyclical indicator for identifying cyclical trends
- **ATR (Average True Range)**: Volatility indicator for position sizing

### System Architecture
- **Modular Design**: Clean separation of concerns with independent modules
- **Failsafe Execution**: Comprehensive error handling and system monitoring
- **Integrity Validation**: CRC32 checksums for all modules and data integrity
- **Auto-reconnection**: Automatic fallback to backup data sources
- **Real-time Alerts**: System-wide alerting and notification system

### Frontend Dashboard
- **Live Charts**: Interactive candlestick charts with Lightweight Charts
- **Real-time Updates**: WebSocket-based live data streaming
- **Responsive Design**: Modern UI built with TailwindCSS
- **Signal Management**: Real-time signal display and management
- **Configuration Panel**: Easy-to-use settings and API key management

## 🏗️ Architecture

### Backend Modules
```
Module 0x01: Data Feed Manager
├── Twelve Data API integration
├── REST polling fallback
├── HTML scraping failsafe
└── Real-time WebSocket streaming

Module 0x02: Divergence Detector
├── Swing high/low detection
├── Multi-indicator analysis
├── Confidence scoring
└── Signal generation

Module 0x03: Signal Engine
├── Binary signal format
├── SQLite logging
├── WebSocket distribution
└── Trade execution hooks

Module 0x04: Frontend Dashboard
├── TailwindCSS styling
├── Lightweight Charts integration
├── Real-time WebSocket client
└── Responsive design

Module 0x05: Integrity Validation
├── CRC32 checksums
├── Module validation
├── Manifest management
└── Error detection

Module 0x06: Docker Deployment
├── Multi-container setup
├── Nginx reverse proxy
├── PostgreSQL database
└── Redis caching

Module 0x07: Failsafe Execution
├── System monitoring
├── Auto-switch fallbacks
├── Critical alerts
└── Trade execution locks
```

## 🛠️ Installation

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for local development)
- Node.js 18+ (for frontend development)
- Twelve Data API key

### Quick Start with Docker

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd forex-trading-bot
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Build and run**
   ```bash
   docker-compose up --build
   ```

4. **Access the application**
   - Frontend: http://localhost
   - Backend API: http://localhost:8000
   - Grafana: http://localhost:3000 (admin/grafana_password_secure)

### Local Development Setup

1. **Backend setup**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python main.py
   ```

2. **Frontend setup**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## 📊 Usage

### 1. API Key Configuration
- Navigate to the dashboard
- Enter your Twelve Data API key
- Test the connection
- Save the configuration

### 2. Monitor Divergences
- View real-time market data on the chart
- Watch for divergence signals in the right panel
- Monitor system alerts and status

### 3. Signal Management
- Review generated trading signals
- Check confidence levels and divergence types
- Monitor signal execution status

### 4. System Configuration
- Adjust confidence thresholds
- Modify update intervals
- Configure auto-execution settings

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# API Configuration
TWELVE_DATA_API_KEY=your_api_key_here
API_RATE_LIMIT=100

# Trading Configuration
TRADING_ENABLED=false
AUTO_EXECUTION=false
CONFIDENCE_THRESHOLD=0.7

# Data Feed Configuration
UPDATE_INTERVAL=1
SYMBOLS=EUR/USD,GBP/USD,USD/JPY,USD/CHF,AUD/USD
```

### Divergence Detection Settings

```python
# Indicator weights
INDICATOR_WEIGHTS_RSI=0.3
INDICATOR_WEIGHTS_MACD=0.25
INDICATOR_WEIGHTS_STOCH=0.25
INDICATOR_WEIGHTS_CCI=0.2

# Detection parameters
LOOKBACK_PERIOD=20
MIN_SWING_DISTANCE=0.001
CONFIDENCE_THRESHOLD=0.7
```

## 📈 API Endpoints

### Core Endpoints
- `GET /` - Application status
- `GET /health` - Health check
- `GET /api/status` - System status
- `POST /api/configure` - Update configuration
- `WebSocket /ws` - Real-time data streaming

### Data Endpoints
- `GET /api/market-data` - Current market data
- `GET /api/signals` - Trading signals
- `GET /api/divergences` - Detected divergences
- `GET /api/alerts` - System alerts

## 🐳 Docker Services

### Core Services
- **backend**: Python FastAPI application
- **frontend**: Nginx with static files
- **database**: PostgreSQL database
- **redis**: Redis caching layer

### Optional Services
- **monitoring**: Prometheus metrics
- **grafana**: Data visualization
- **database-init**: Database initialization

### Ports
- Frontend: 80
- Backend: 8000
- Database: 5432
- Redis: 6379
- Prometheus: 9090
- Grafana: 3000

## 🔒 Security Features

- **API Key Encryption**: Secure storage with Fernet encryption
- **Input Validation**: Comprehensive data validation
- **Rate Limiting**: API rate limiting and throttling
- **CORS Protection**: Configurable cross-origin policies
- **Security Headers**: Modern security headers implementation

## 📊 Monitoring and Logging

### System Monitoring
- Real-time system status
- Connection health monitoring
- Performance metrics
- Error tracking and alerting

### Logging
- Structured logging with different levels
- Log rotation and archival
- Centralized log management
- Performance and debugging logs

## 🚨 Failsafe Features

### Automatic Fallbacks
- Primary data source failure → REST API fallback
- REST API failure → HTML scraping fallback
- Connection monitoring and auto-reconnection
- Graceful degradation handling

### Trade Protection
- System health monitoring
- Automatic trade execution locks
- Critical condition alerts
- Emergency shutdown procedures

## 🧪 Testing

### Backend Tests
```bash
cd backend
python -m pytest tests/
```

### Frontend Tests
```bash
cd frontend
npm test
```

### Integration Tests
```bash
docker-compose -f docker-compose.test.yml up --build
```

## 📚 API Documentation

Once running, access the interactive API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading forex involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making trading decisions.

## 🆘 Support

- **Issues**: GitHub Issues
- **Documentation**: This README and inline code comments
- **Community**: GitHub Discussions

## 🔄 Changelog

### Version 1.0.0
- Initial release with all core modules
- Complete divergence detection system
- Real-time dashboard
- Docker deployment support
- Comprehensive failsafe mechanisms

---

**Built with ❤️ using modern web technologies and best practices**
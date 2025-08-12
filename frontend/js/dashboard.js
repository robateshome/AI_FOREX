/**
 * Forex Trading Bot Dashboard
 * Module 0x04: Frontend Dashboard JavaScript
 * CRC32: PLACEHOLDER_CRC32_04_JS
 */

class ForexDashboard {
    constructor() {
        this.websocket = null;
        this.chart = null;
        this.candlestickSeries = null;
        this.volumeSeries = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        
        // Data storage
        this.marketData = {};
        this.signals = [];
        this.divergences = [];
        this.alerts = [];
        
        // Configuration
        this.config = {
            apiKey: '',
            confidenceThreshold: 0.7,
            updateInterval: 1,
            autoExecution: false,
            currentTimeframe: '5m'
        };
        
        this.init();
    }
    
    init() {
        this.loadConfiguration();
        this.initializeChart();
        this.setupEventListeners();
        this.connectWebSocket();
        this.startDataPolling();
    }
    
    loadConfiguration() {
        // Load saved configuration from localStorage
        const savedConfig = localStorage.getItem('forexBotConfig');
        if (savedConfig) {
            this.config = { ...this.config, ...JSON.parse(savedConfig) };
        }
        
        // Load API key if available
        const savedApiKey = localStorage.getItem('forexBotApiKey');
        if (savedApiKey) {
            this.config.apiKey = savedApiKey;
            document.getElementById('apiKeyInput').value = savedApiKey;
        }
        
        // Update UI with loaded config
        this.updateConfigurationUI();
    }
    
    saveConfiguration() {
        localStorage.setItem('forexBotConfig', JSON.stringify(this.config));
        if (this.config.apiKey) {
            localStorage.setItem('forexBotApiKey', this.config.apiKey);
        }
    }
    
    updateConfigurationUI() {
        document.getElementById('confidenceThreshold').value = this.config.confidenceThreshold;
        document.getElementById('confidenceValue').textContent = this.config.confidenceThreshold;
        document.getElementById('updateInterval').value = this.config.updateInterval;
        document.getElementById('autoExecution').checked = this.config.autoExecution;
    }
    
    initializeChart() {
        const chartContainer = document.getElementById('chartContainer');
        
        // Create chart
        this.chart = LightweightCharts.createChart(chartContainer, {
            width: chartContainer.clientWidth,
            height: chartContainer.clientHeight,
            layout: {
                background: { color: '#1F2937' },
                textColor: '#D1D5DB',
            },
            grid: {
                vertLines: { color: '#374151' },
                horzLines: { color: '#374151' },
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
            },
            rightPriceScale: {
                borderColor: '#374151',
            },
            timeScale: {
                borderColor: '#374151',
                timeVisible: true,
                secondsVisible: false,
            },
        });
        
        // Create candlestick series
        this.candlestickSeries = this.chart.addCandlestickSeries({
            upColor: '#10B981',
            downColor: '#EF4444',
            borderDownColor: '#EF4444',
            borderUpColor: '#10B981',
            wickDownColor: '#EF4444',
            wickUpColor: '#10B981',
        });
        
        // Create volume series
        this.volumeSeries = this.chart.addHistogramSeries({
            color: '#3B82F6',
            priceFormat: {
                type: 'volume',
            },
            priceScaleId: '',
            scaleMargins: {
                top: 0.8,
                bottom: 0,
            },
        });
        
        // Handle window resize
        window.addEventListener('resize', () => {
            this.chart.applyOptions({
                width: chartContainer.clientWidth,
                height: chartContainer.clientHeight,
            });
        });
        
        // Add sample data for demonstration
        this.addSampleData();
    }
    
    addSampleData() {
        // Add sample candlestick data
        const sampleData = [
            { time: '2024-01-01', open: 1.1000, high: 1.1050, low: 1.0980, close: 1.1030 },
            { time: '2024-01-02', open: 1.1030, high: 1.1080, low: 1.1010, close: 1.1060 },
            { time: '2024-01-03', open: 1.1060, high: 1.1100, low: 1.1040, close: 1.1080 },
            { time: '2024-01-04', open: 1.1080, high: 1.1120, low: 1.1060, close: 1.1090 },
            { time: '2024-01-05', open: 1.1090, high: 1.1130, low: 1.1070, close: 1.1110 },
        ];
        
        this.candlestickSeries.setData(sampleData);
        
        // Add sample volume data
        const sampleVolume = sampleData.map((candle, index) => ({
            time: candle.time,
            value: Math.random() * 1000000 + 500000,
            color: candle.close >= candle.open ? '#10B981' : '#EF4444',
        }));
        
        this.volumeSeries.setData(sampleVolume);
    }
    
    setupEventListeners() {
        // API Key management
        document.getElementById('saveApiKeyBtn').addEventListener('click', () => {
            this.saveApiKey();
        });
        
        document.getElementById('testConnectionBtn').addEventListener('click', () => {
            this.testConnection();
        });
        
        // Settings modal
        document.getElementById('settingsBtn').addEventListener('click', () => {
            this.showSettingsModal();
        });
        
        document.getElementById('closeSettingsBtn').addEventListener('click', () => {
            this.hideSettingsModal();
        });
        
        document.getElementById('saveSettingsBtn').addEventListener('click', () => {
            this.saveSettings();
        });
        
        // Configuration inputs
        document.getElementById('confidenceThreshold').addEventListener('input', (e) => {
            document.getElementById('confidenceValue').textContent = e.target.value;
        });
        
        // Timeframe buttons
        document.querySelectorAll('[data-timeframe]').forEach(button => {
            button.addEventListener('click', (e) => {
                this.changeTimeframe(e.target.dataset.timeframe);
            });
        });
    }
    
    saveApiKey() {
        const apiKey = document.getElementById('apiKeyInput').value.trim();
        if (apiKey) {
            this.config.apiKey = apiKey;
            this.saveConfiguration();
            this.showNotification('API key saved successfully', 'success');
            
            // Test connection automatically
            this.testConnection();
        } else {
            this.showNotification('Please enter a valid API key', 'error');
        }
    }
    
    async testConnection() {
        if (!this.config.apiKey) {
            this.showNotification('Please save an API key first', 'error');
            return;
        }
        
        try {
            const response = await fetch('/api/configure', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    data_feed: { api_key: this.config.apiKey }
                }),
            });
            
            if (response.ok) {
                this.showNotification('Connection test successful', 'success');
                this.updateConnectionStatus('twelveData', 'connected');
            } else {
                this.showNotification('Connection test failed', 'error');
                this.updateConnectionStatus('twelveData', 'error');
            }
        } catch (error) {
            console.error('Connection test error:', error);
            this.showNotification('Connection test failed', 'error');
            this.updateConnectionStatus('twelveData', 'error');
        }
    }
    
    showSettingsModal() {
        document.getElementById('settingsModal').classList.remove('hidden');
    }
    
    hideSettingsModal() {
        document.getElementById('settingsModal').classList.add('hidden');
    }
    
    saveSettings() {
        this.config.confidenceThreshold = parseFloat(document.getElementById('confidenceThreshold').value);
        this.config.updateInterval = parseInt(document.getElementById('updateInterval').value);
        this.config.autoExecution = document.getElementById('autoExecution').checked;
        
        this.saveConfiguration();
        this.showNotification('Settings saved successfully', 'success');
        this.hideSettingsModal();
        
        // Send configuration to backend
        this.sendConfiguration();
    }
    
    async sendConfiguration() {
        try {
            const response = await fetch('/api/configure', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    divergence: { confidence_threshold: this.config.confidenceThreshold },
                    data_feed: { update_interval: this.config.updateInterval }
                }),
            });
            
            if (response.ok) {
                console.log('Configuration sent to backend');
            }
        } catch (error) {
            console.error('Error sending configuration:', error);
        }
    }
    
    changeTimeframe(timeframe) {
        // Update active timeframe button
        document.querySelectorAll('[data-timeframe]').forEach(button => {
            button.classList.remove('bg-forex-blue');
            button.classList.add('bg-gray-700');
        });
        
        document.querySelector(`[data-timeframe="${timeframe}"]`).classList.remove('bg-gray-700');
        document.querySelector(`[data-timeframe="${timeframe}"]`).classList.add('bg-forex-blue');
        
        this.config.currentTimeframe = timeframe;
        
        // In a real implementation, this would request new data for the timeframe
        console.log(`Timeframe changed to ${timeframe}`);
    }
    
    connectWebSocket() {
        try {
            this.websocket = new WebSocket(`ws://${window.location.host}/ws`);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus('websocket', 'connected');
                this.showNotification('WebSocket connected', 'success');
            };
            
            this.websocket.onmessage = (event) => {
                this.handleWebSocketMessage(event.data);
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.isConnected = false;
                this.updateConnectionStatus('websocket', 'disconnected');
                this.showNotification('WebSocket disconnected', 'warning');
                
                // Attempt reconnection
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    setTimeout(() => {
                        this.reconnectAttempts++;
                        this.connectWebSocket();
                    }, 5000);
                }
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('websocket', 'error');
            };
            
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.updateConnectionStatus('websocket', 'error');
        }
    }
    
    handleWebSocketMessage(data) {
        try {
            const message = JSON.parse(data);
            
            switch (message.type) {
                case 'market_data':
                    this.updateMarketData(message.data);
                    break;
                case 'signal':
                    this.addSignal(message.data);
                    break;
                case 'divergence':
                    this.addDivergence(message.data);
                    break;
                case 'alert':
                    this.addAlert(message.data);
                    break;
                case 'status':
                    this.updateSystemStatus(message.data);
                    break;
                default:
                    console.log('Unknown message type:', message.type);
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    }
    
    updateMarketData(data) {
        // Update market data storage
        this.marketData = { ...this.marketData, ...data };
        
        // Update chart with new data
        Object.entries(data).forEach(([symbol, marketData]) => {
            if (marketData.close) {
                const candleData = {
                    time: Math.floor(Date.now() / 1000),
                    open: marketData.open,
                    high: marketData.high,
                    low: marketData.low,
                    close: marketData.close,
                };
                
                this.candlestickSeries.update(candleData);
                
                // Update volume if available
                if (marketData.volume) {
                    const volumeData = {
                        time: Math.floor(Date.now() / 1000),
                        value: marketData.volume,
                        color: marketData.close >= marketData.open ? '#10B981' : '#EF4444',
                    };
                    
                    this.volumeSeries.update(volumeData);
                }
            }
        });
        
        // Update status displays
        this.updateDataSourceDisplay();
    }
    
    addSignal(signal) {
        // Add to signals list
        this.signals.unshift(signal);
        if (this.signals.length > 50) {
            this.signals.pop();
        }
        
        // Update UI
        this.updateSignalsList();
        this.updateActiveSignalsCount();
        
        // Show notification
        this.showNotification(`New ${signal.signal_type} signal for ${signal.symbol}`, 'info');
        
        // Add divergence marker to chart if applicable
        if (signal.divergence_type !== 'none') {
            this.addDivergenceMarker(signal);
        }
    }
    
    addDivergence(divergence) {
        // Add to divergences list
        this.divergences.unshift(divergence);
        if (this.divergences.length > 20) {
            this.divergences.pop();
        }
        
        // Update UI
        this.updateDivergenceList();
        
        // Show notification
        this.showNotification(`${divergence.divergence_type} divergence detected for ${divergence.symbol}`, 'warning');
    }
    
    addAlert(alert) {
        // Add to alerts list
        this.alerts.unshift(alert);
        if (this.alerts.length > 20) {
            this.alerts.pop();
        }
        
        // Update UI
        this.updateAlertsList();
        
        // Show notification based on alert level
        const notificationType = this.getNotificationType(alert.level);
        this.showNotification(alert.message, notificationType);
    }
    
    updateSystemStatus(status) {
        // Update system status display
        document.getElementById('systemStatus').textContent = status.status;
        document.getElementById('systemStatus').className = `text-lg font-semibold ${this.getStatusColor(status.status)}`;
        
        // Update trade lock status
        if (status.trade_locked !== undefined) {
            document.getElementById('tradeLockStatus').textContent = status.trade_locked ? 'Locked' : 'Unlocked';
            document.getElementById('tradeLockStatus').className = `text-lg font-semibold ${status.trade_locked ? 'text-forex-red' : 'text-forex-green'}`;
        }
        
        // Update connection statuses
        if (status.connection_status) {
            Object.entries(status.connection_status).forEach(([source, status]) => {
                this.updateConnectionStatus(source, status);
            });
        }
    }
    
    updateSignalsList() {
        const signalsList = document.getElementById('signalsList');
        
        if (this.signals.length === 0) {
            signalsList.innerHTML = `
                <div class="text-center text-gray-500 py-8">
                    <p>No signals yet</p>
                    <p class="text-sm">Signals will appear here when detected</p>
                </div>
            `;
            return;
        }
        
        signalsList.innerHTML = this.signals.map(signal => `
            <div class="bg-gray-800 p-3 rounded-lg border-l-4 ${this.getSignalBorderColor(signal.signal_type)}">
                <div class="flex justify-between items-start">
                    <div>
                        <div class="font-medium">${signal.symbol}</div>
                        <div class="text-sm text-gray-400">${signal.signal_type}</div>
                        <div class="text-xs text-gray-500">${signal.divergence_type}</div>
                    </div>
                    <div class="text-right">
                        <div class="text-sm font-medium">${signal.price}</div>
                        <div class="text-xs text-gray-400">${(signal.confidence_level * 100).toFixed(1)}%</div>
                        <div class="text-xs text-gray-500">${new Date(signal.timestamp).toLocaleTimeString()}</div>
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    updateDivergenceList() {
        const divergenceList = document.getElementById('divergenceList');
        
        if (this.divergences.length === 0) {
            divergenceList.innerHTML = `
                <div class="text-center text-gray-500 py-8">
                    <p>No divergences detected</p>
                    <p class="text-sm">Monitoring for bullish/bearish divergences</p>
                </div>
            `;
            return;
        }
        
        divergenceList.innerHTML = this.divergences.map(divergence => `
            <div class="bg-gray-800 p-3 rounded-lg border-l-4 ${this.getDivergenceBorderColor(divergence.divergence_type)}">
                <div class="flex justify-between items-start">
                    <div>
                        <div class="font-medium">${divergence.symbol}</div>
                        <div class="text-sm text-gray-400">${divergence.divergence_type}</div>
                        <div class="text-xs text-gray-500">${divergence.description}</div>
                    </div>
                    <div class="text-right">
                        <div class="text-sm font-medium">${divergence.price}</div>
                        <div class="text-xs text-gray-400">${(divergence.confidence_level * 100).toFixed(1)}%</div>
                        <div class="text-xs text-gray-500">${new Date(divergence.timestamp).toLocaleTimeString()}</div>
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    updateAlertsList() {
        const alertsList = document.getElementById('alertsList');
        
        if (this.alerts.length === 0) {
            alertsList.innerHTML = `
                <div class="text-center text-gray-500 py-8">
                    <p>No alerts</p>
                    <p class="text-sm">System is running normally</p>
                </div>
            `;
            return;
        }
        
        alertsList.innerHTML = this.alerts.map(alert => `
            <div class="bg-gray-800 p-3 rounded-lg border-l-4 ${this.getAlertBorderColor(alert.level)}">
                <div class="flex justify-between items-start">
                    <div>
                        <div class="font-medium">${alert.module}</div>
                        <div class="text-sm text-gray-400">${alert.message}</div>
                    </div>
                    <div class="text-right">
                        <div class="text-xs text-gray-500">${new Date(alert.timestamp).toLocaleTimeString()}</div>
                        <div class="text-xs ${alert.acknowledged ? 'text-forex-green' : 'text-forex-yellow'}">
                            ${alert.acknowledged ? 'Acknowledged' : 'Pending'}
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    updateActiveSignalsCount() {
        const activeCount = this.signals.filter(signal => signal.status === 'pending').length;
        document.getElementById('activeSignals').textContent = activeCount;
    }
    
    updateDataSourceDisplay() {
        // This would update based on actual data source status
        document.getElementById('dataSource').textContent = 'Twelve Data';
    }
    
    updateConnectionStatus(source, status) {
        const statusElement = document.getElementById(`${source}Status`);
        if (statusElement) {
            statusElement.className = `w-3 h-3 rounded-full ${this.getConnectionStatusColor(status)}`;
        }
    }
    
    addDivergenceMarker(signal) {
        // Add visual marker to chart for divergence
        const marker = {
            time: Math.floor(Date.now() / 1000),
            position: signal.signal_type === 'BUY' ? 'belowBar' : 'aboveBar',
            color: signal.signal_type === 'BUY' ? '#10B981' : '#EF4444',
            shape: signal.signal_type === 'BUY' ? 'arrowUp' : 'arrowDown',
            text: `${signal.divergence_type} divergence`,
        };
        
        // In a real implementation, you would add this marker to the chart
        console.log('Adding divergence marker:', marker);
    }
    
    startDataPolling() {
        // Poll for data updates as fallback to WebSocket
        setInterval(async () => {
            if (!this.isConnected) {
                await this.pollData();
            }
        }, this.config.updateInterval * 1000);
    }
    
    async pollData() {
        try {
            const response = await fetch('/api/status');
            if (response.ok) {
                const data = await response.json();
                this.updateSystemStatus(data);
            }
        } catch (error) {
            console.error('Data polling error:', error);
        }
    }
    
    // Utility methods
    getSignalBorderColor(signalType) {
        switch (signalType) {
            case 'BUY': return 'border-forex-green';
            case 'SELL': return 'border-forex-red';
            case 'HOLD': return 'border-forex-yellow';
            default: return 'border-gray-600';
        }
    }
    
    getDivergenceBorderColor(divergenceType) {
        if (divergenceType.includes('bullish')) return 'border-forex-green';
        if (divergenceType.includes('bearish')) return 'border-forex-red';
        return 'border-gray-600';
    }
    
    getAlertBorderColor(level) {
        switch (level) {
            case 'critical': return 'border-forex-red';
            case 'error': return 'border-forex-red';
            case 'warning': return 'border-forex-yellow';
            case 'info': return 'border-forex-blue';
            default: return 'border-gray-600';
        }
    }
    
    getStatusColor(status) {
        switch (status) {
            case 'operational': return 'text-forex-green';
            case 'degraded': return 'text-forex-yellow';
            case 'critical': return 'text-forex-red';
            case 'emergency': return 'text-red-600';
            default: return 'text-gray-400';
        }
    }
    
    getConnectionStatusColor(status) {
        switch (status) {
            case 'connected': return 'bg-forex-green';
            case 'connecting': return 'bg-forex-yellow';
            case 'error': return 'bg-forex-red';
            case 'disconnected': return 'bg-gray-500';
            default: return 'bg-gray-500';
        }
    }
    
    getNotificationType(alertLevel) {
        switch (alertLevel) {
            case 'critical': return 'error';
            case 'error': return 'error';
            case 'warning': return 'warning';
            case 'info': return 'info';
            default: return 'info';
        }
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 ${this.getNotificationClasses(type)}`;
        notification.innerHTML = `
            <div class="flex items-center">
                <span class="mr-2">${this.getNotificationIcon(type)}</span>
                <span>${message}</span>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }
    
    getNotificationClasses(type) {
        switch (type) {
            case 'success': return 'bg-forex-green text-white';
            case 'error': return 'bg-forex-red text-white';
            case 'warning': return 'bg-forex-yellow text-black';
            case 'info': return 'bg-forex-blue text-white';
            default: return 'bg-gray-600 text-white';
        }
    }
    
    getNotificationIcon(type) {
        switch (type) {
            case 'success': return '✓';
            case 'error': return '✗';
            case 'warning': return '⚠';
            case 'info': return 'ℹ';
            default: return '•';
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.forexDashboard = new ForexDashboard();
});
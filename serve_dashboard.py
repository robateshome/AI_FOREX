#!/usr/bin/env python3
"""
Simple web server to serve the AI Forex Signal Generator Dashboard
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

# Configuration
PORT = 8081
FRONTEND_DIR = Path("frontend")

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    # Ensure frontend directory exists
    if not FRONTEND_DIR.exists():
        print(f"❌ Frontend directory not found: {FRONTEND_DIR}")
        print("Creating frontend directory...")
        FRONTEND_DIR.mkdir(exist_ok=True)
    
    # Check if index.html exists
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        print(f"❌ Dashboard file not found: {index_file}")
        return
    
    # Start server
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"🚀 AI Forex Signal Generator Dashboard")
        print(f"🌐 Web Interface: http://localhost:{PORT}")
        print(f"🔧 API Backend: http://localhost:8000")
        print(f"📁 Serving from: {FRONTEND_DIR.absolute()}")
        print(f"🔄 Auto-refresh: Every 5 seconds")
        print(f"\n💡 Open your browser and navigate to: http://localhost:{PORT}")
        print(f"⏹️  Press Ctrl+C to stop the server")
        
        # Try to open browser automatically
        try:
            webbrowser.open(f"http://localhost:{PORT}")
            print(f"🌐 Browser opened automatically!")
        except:
            print(f"🌐 Please open your browser manually")
        
        print(f"\n✅ Dashboard server started successfully!")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"\n⏹️  Dashboard server stopped")

if __name__ == "__main__":
    main()
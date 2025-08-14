#!/usr/bin/env python3
"""
Simple test script to verify Twelve Data API connection
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_twelve_data_api():
    """Test the Twelve Data API connection"""
    try:
        from modules.twelve_data_client import create_twelve_data_client
        
        print("🔑 Testing Twelve Data API connection...")
        
        # Create client
        client = create_twelve_data_client()
        
        # Test real-time quote
        print("📊 Testing real-time quote...")
        quote = await client.get_real_time_quote("EUR/USD")
        print(f"✅ Quote received: {quote['symbol']} = {quote['close']}")
        
        # Test historical data
        print("📈 Testing historical data...")
        historical = await client.get_historical_data("EUR/USD", "1min", 100)
        print(f"✅ Historical data received: {len(historical)} records")
        
        print("🎉 All API tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

async def test_ai_model():
    """Test AI model creation"""
    try:
        from modules.ai_signal_generator import AISignalGenerator
        
        print("🤖 Testing AI model creation...")
        
        # Create AI generator
        ai_gen = AISignalGenerator()
        print("✅ AI model created successfully")
        
        print("🎉 AI model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ AI model test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🧪 Running API and AI tests...")
    print("=" * 50)
    
    # Test API
    api_ok = await test_twelve_data_api()
    
    # Test AI model
    ai_ok = await test_ai_model()
    
    print("=" * 50)
    if api_ok and ai_ok:
        print("🎉 All tests passed! Ready to start the server.")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    asyncio.run(main())
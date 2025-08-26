#!/usr/bin/env python3
"""
Simple test script to verify Monica Bot modules work correctly
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_modules():
    """Test that all modules can be imported and basic functions work."""
    print("Testing Monica Bot modules...")
    
    try:
        # Test config module
        from modules.config import load_config, DEFAULT_CONFIG
        config = load_config()
        print("✅ Config module: OK")
        print(f"   Default prefix: {DEFAULT_CONFIG['prefix']}")
        print(f"   Loaded config keys: {list(config.keys())}")
    except Exception as e:
        print(f"❌ Config module: FAILED - {e}")
        return False
    
    try:
        # Test utils module
        from modules.utils import format_duration, truncate, THEME_COLOR
        duration_test = format_duration(3661)  # 1h 1m 1s
        truncate_test = truncate("This is a very long string that should be truncated", 20)
        print("✅ Utils module: OK")
        print(f"   Duration test (3661s): {duration_test}")
        print(f"   Truncate test: {truncate_test}")
        print(f"   Theme color: {hex(THEME_COLOR)}")
    except Exception as e:
        print(f"❌ Utils module: FAILED - {e}")
        return False
    
    try:
        # Test metrics module
        from modules.metrics import metric_inc, metrics_snapshot, get_average_resolve_time
        metric_inc("test_metric", 5)
        snapshot = metrics_snapshot()
        avg_time = get_average_resolve_time()
        print("✅ Metrics module: OK")
        print(f"   Test metric value: {snapshot.get('test_metric', 0)}")
        print(f"   Average resolve time: {avg_time:.3f}s")
    except Exception as e:
        print(f"❌ Metrics module: FAILED - {e}")
        return False
    
    try:
        # Test exceptions module
        from modules.exceptions import MonicaBotError, ConfigurationError, ResolveError
        print("✅ Exceptions module: OK")
        print(f"   Available exceptions: MonicaBotError, ConfigurationError, ResolveError, etc.")
    except Exception as e:
        print(f"❌ Exceptions module: FAILED - {e}")
        return False
    
    print("\n🎉 All modules loaded successfully!")
    return True

def test_token_security():
    """Test that token is properly handled via environment variables."""
    print("\nTesting token security...")
    
    try:
        from modules.config import get_token
        # This should fail if no DISCORD_TOKEN is set
        token = get_token()
        if token and len(token) > 20:
            print("✅ Token found in environment")
            print(f"   Token length: {len(token)} characters")
            print(f"   Token preview: {token[:10]}...{token[-4:]}")
        else:
            print("⚠️  No valid token found - this is expected in test environment")
    except ValueError as e:
        print(f"⚠️  Token security test: {e}")
        print("   This is expected if DISCORD_TOKEN environment variable is not set")
    except Exception as e:
        print(f"❌ Token security test: FAILED - {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Monica Bot v3.5.2 - Module Test Script")
    print("=" * 50)
    
    success = test_modules()
    if success:
        test_token_security()
        print("\n✨ All tests completed!")
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)

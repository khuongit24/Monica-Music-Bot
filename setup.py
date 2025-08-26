#!/usr/bin/env python3
"""
Monica Bot Setup Script
Helps users set up the bot environment correctly
"""

import os
import sys
import subprocess
import json

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required!")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_ffmpeg():
    """Check if FFmpeg is installed."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg is installed")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("⚠️  FFmpeg not found!")
    print("   Please install FFmpeg from https://ffmpeg.org/download.html")
    return False

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ['discord.py', 'yt-dlp', 'PyNaCl']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').replace('.py', ''))
            print(f"✅ {package} is installed")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} is missing")
    
    if missing:
        print(f"\n📦 To install missing packages, run:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True

def setup_env_file():
    """Help user set up .env file."""
    if os.path.exists('.env'):
        print("✅ .env file already exists")
        return True
    
    if not os.path.exists('.env.example'):
        print("❌ .env.example file not found!")
        return False
    
    print("\n🔧 Setting up .env file...")
    token = input("Enter your Discord Bot Token (or press Enter to skip): ").strip()
    
    try:
        with open('.env.example', 'r', encoding='utf-8') as f:
            content = f.read()
        
        if token:
            content = content.replace('your_discord_bot_token_here', token)
        
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ .env file created!")
        if not token:
            print("⚠️  Don't forget to edit .env and add your Discord Bot Token!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def validate_config():
    """Validate config.json file."""
    if not os.path.exists('config.json'):
        print("⚠️  config.json not found, will use defaults")
        return True
    
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Check for token in config (security issue)
        if 'token' in config:
            print("⚠️  WARNING: Token found in config.json!")
            print("   This is a security risk. Token should be in .env file only.")
            
            remove = input("Remove token from config.json? (y/N): ").strip().lower()
            if remove == 'y':
                del config['token']
                with open('config.json', 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
                print("✅ Token removed from config.json")
        
        print("✅ config.json is valid")
        return True
        
    except Exception as e:
        print(f"❌ config.json validation failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🎵 Monica Discord Music Bot v3.4.3 - Setup Script")
    print("=" * 60)
    
    all_good = True
    
    print("\n1️⃣ Checking Python version...")
    if not check_python_version():
        all_good = False
    
    print("\n2️⃣ Checking FFmpeg...")
    if not check_ffmpeg():
        all_good = False
    
    print("\n3️⃣ Checking Python dependencies...")
    if not check_dependencies():
        all_good = False
    
    print("\n4️⃣ Setting up environment file...")
    if not setup_env_file():
        all_good = False
    
    print("\n5️⃣ Validating configuration...")
    if not validate_config():
        all_good = False
    
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 Setup completed successfully!")
        print("\n📝 Next steps:")
        print("   1. Make sure you have a Discord Bot Token in your .env file")
        print("   2. Run: python bot.py")
        print("   3. Invite the bot to your Discord server")
        print("\n📚 For more help, check README.md")
    else:
        print("⚠️  Setup completed with warnings.")
        print("   Please fix the issues above before running the bot.")
    
    print("\n💡 Tip: Run 'python test_modules.py' to test the installation")

if __name__ == "__main__":
    main()

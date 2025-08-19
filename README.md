# Discord Music Bot (ver_1.0_release)

Bot discord dùng để phát nhạc.

## Tính năng chính
- Hỗ trợ Slash commands và fallback prefix commands
- Hệ thống hàng đợi (per-guild)
- Lưu playlist persist vào `playlists.json`
- Các lệnh phổ biến: join, leave, play, pause, resume, stop, skip, queue, now, volume, list_playlists, save_playlist, play_playlist, shutdown (owner only)
- Dockerfile + docker-compose để triển khai nhanh trên VPS

## Yêu cầu
- Python 3.10+ (khuyến nghị 3.11)
- ffmpeg (trong Dockerfile đã cài sẵn nếu dùng Docker image)
- Token bot Discord (tạo trên Developer Portal) và quyền: Connect, Speak, Send Messages, Use Application Commands

## Cài đặt nhanh (cục bộ)
1. Tạo virtualenv
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate    # Windows PowerShell

2. Cài dependencies
   pip install -r requirements.txt

3. Tạo file cấu hình
   Sao chép config.json.example thành config.json và điền token (và owner_id nếu muốn):
   {
     "token": "YOUR_TOKEN",
     "prefix": "!",
     "owner_id": 123456789012345678
   }

4. Chạy bot
   python bot.py

## Chạy bằng Docker
1. Thiết lập biến môi trường DISCORD_TOKEN trên host
2. Xây dựng và chạy:
   docker compose up -d --build

## Lưu ý
- Free Source, không sử dụng với mục đích thương mại, không chịu trách nhiệm trước bất kỳ pháp lý nào.

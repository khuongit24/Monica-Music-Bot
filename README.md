# Discord Music Bot (ver_1.0_release)

Đây là phiên bản chính thức ver_1.0_release cho Discord Music Bot, hoàn thiện hơn so với bản pre_release.
Mục đích: sử dụng cá nhân trong server riêng tư (không thương mại).

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

## Chạy bằng Docker (khuyến nghị cho sản phẩm ổn định)
1. Thiết lập biến môi trường DISCORD_TOKEN trên host
2. Xây dựng và chạy:
   docker compose up -d --build

## Lưu ý pháp lý và đạo đức
- Mặc dù bạn cho biết dùng cho server riêng tư, hãy nhớ rằng việc stream nội dung có bản quyền từ các nguồn công cộng có thể chịu các quy định dịch vụ của nền tảng (ví dụ YouTube).
- Bot này cung cấp công cụ kỹ thuật để phát luồng audio; trách nhiệm tuân thủ luật bản quyền thuộc về người dùng.

## Tùy chỉnh & mở rộng (gợi ý)
- Thêm chế độ shuffle, loop, repeat, hoặc quyền chỉ cho một role dùng lệnh quản lý.
- Thêm cơ chế cache/local download để giảm tải network (cân nhắc bản quyền).
- Chuyển toàn bộ logic thành cogs để bảo trì dễ dàng.

## Hỗ trợ
Mình đóng vai lập trình viên 10 năm — nếu cần mình bổ sung cogs, hệ thống DB (SQLite/Postgres) cho playlists, hoặc UI web để điều khiển, mình làm tiếp.
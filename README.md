# Monica Discord Music Bot v2.1 - Free Source - Free To Use

Bot discord dùng để phát nhạc.

## Yêu cầu
- Đã cài sẵn Python 3.10+ (khuyến nghị 3.11)
- Đã cài ffmpeg (trong Dockerfile đã cài sẵn nếu dùng Docker image). Bạn không biết cài thì tìm google cách cài hai thứ này nhé. Dễ lắm 👌
- Token bot Discord (tạo trên Developer Application Portal) và tạo Bot trên đó. Tương tự như trên, bạn không biết thì xem youtube cách tạo app trên discord portal nhé, sau đó quay lại đây

## Cài đặt nhanh (cục bộ)
1. Chỉnh file config.json và điền token(Token Discord Application - bắt buộc, không có thì bot không chạy được đâu):
   {"token": "Điền token app của bạn vào đây", "prefix": "!", }
2. Cài dependencies pip install -r requirements.txt
3. Mở Terminal gõ cd + đường dẫn chứa bot
4. Gõ python bot.py và enter chạy thôi


## Chạy bằng Docker
1. Thiết lập biến môi trường DISCORD_TOKEN trên host
2. Xây dựng và chạy:
   docker compose up -d --build

## Lưu ý
- Free Source, không sử dụng với mục đích thương mại, không chịu trách nhiệm trước bất kỳ pháp lý nào.

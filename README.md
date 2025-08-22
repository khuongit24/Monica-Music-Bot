# Monica Discord Music Bot v3.4.2

Đây là một dự án bot nhạc cho Discord được phát triển **dành cho mục đích sử dụng cá nhân và nghiên cứu**.

![Use at your own risk](https://img.shields.io/badge/Disclaimer-Use%20at%20your%20own%20risk-red)  
![Non-commercial](https://img.shields.io/badge/Use-Non%20Commercial-blue)

---

## Mục đích dự án

Monica được phát triển nhằm cung cấp một mẫu tham khảo mã nguồn mở giúp người dùng hiểu cách xây dựng và vận hành một Discord music bot. Dự án hướng tới mục tiêu giáo dục, thử nghiệm và sử dụng cá nhân — **không** nhằm mục đích thương mại hoặc phân phối đại trà.

## Tuyên bố miễn trừ trách nhiệm

Dự án này (Monica-Music-Bot) được cung cấp **chỉ cho mục đích cá nhân và giáo dục**. Nó không có sự liên kết, bảo trợ hay ủy quyền từ YouTube, Google hoặc bất kỳ nhà cung cấp nội dung nào khác.

Khi sử dụng phần mềm này, bạn đồng ý và hiểu rõ rằng:

- Bạn chịu toàn bộ trách nhiệm về cách thức sử dụng bot.
- Tác giả và người đóng góp **không chịu trách nhiệm pháp lý** cho bất kỳ hành vi sử dụng sai mục đích nào của mã nguồn, bao gồm nhưng không giới hạn ở việc vi phạm bản quyền hoặc vi phạm Điều khoản Dịch vụ của các nền tảng bên thứ ba.
- Dự án này **không được phép** sử dụng cho mục đích thương mại, phân phối đại trà, hoặc bất kỳ hoạt động nào xâm phạm quyền sở hữu trí tuệ.

Việc sử dụng phần mềm để tải xuống, phát trực tuyến hoặc phân phối nội dung có bản quyền khi không có sự cho phép **có thể vi phạm pháp luật và Điều khoản Dịch vụ** của các nền tảng liên quan. Người dùng phải tự chịu trách nhiệm về mọi rủi ro pháp lý và tuân thủ các quy định hiện hành.

---

## Tính năng chính

Phát nhạc
/join • /play <query> • /pause • /resume • /skip • /stop • /leave
Hàng đợi
/queue • /clear <tên> • /clear_all • /reverse
Loop / Lịch sử
/loop (loop 1 bài) • /loop_all (loop hàng đợi) • /unloop (tắt cả hai) • /reverse
Thông tin / Giám sát
/now • /stats • /health • /metrics • /version
Cấu hình / Debug
/profile • /volume • /debug_track <query> • /config_show
Báo cáo
/report (hoặc !report) để mở form gửi lỗi / góp ý
Nguồn hỗ trợ
YouTube • SoundCloud • Bandcamp • Mixcloud • Audius


## Yêu cầu hệ thống

- Python 3.8 hoặc mới hơn.  
- FFmpeg đã cài đặt và có trong PATH.  
- Thư viện phụ thuộc (tham khảo `requirements.txt`).

## Cài đặt (hướng dẫn cơ bản)

1. Tải về file zip của bot
2. Cài đặt phụ thuộc: `pip install -r requirements.txt`.
3. Cấu hình cài token discord application của bạn trong config.json
4. Mở terminal và cd sang thư mục chứa bot. Ví dụ cd D:\Monica_Bot
4. Gõ lệnh chạy bot: `python bot.py`.

## Sử dụng

- Vui lòng tuân thủ hướng dẫn sử dụng và các lệnh được cung cấp trong repository.  
- Tránh sử dụng bot để phân phối nội dung trái phép hoặc phục vụ mục đích thương mại khi không có quyền hợp lệ.

## Đóng góp

Mọi đóng góp đều được hoan nghênh trong giới hạn pháp lý và mục đích giáo dục. Nếu bạn muốn góp ý hoặc gửi pull request, vui lòng:

1. Mở issue mô tả vấn đề hoặc tính năng.  
2. Tạo pull request kèm theo mô tả và test case (nếu cần).  

Người duyệt sẽ xem xét và phản hồi theo tiêu chuẩn chung của dự án.

## Giấy phép

Vui lòng kiểm tra file `LICENSE` trong repository để biết chi tiết về giấy phép. Dựa trên tuyên bố trong README, dự án này **không** nên được dùng cho mục đích thương mại hoặc phân phối đại trà nếu điều đó mâu thuẫn với giấy phép thực tế. Trong trường hợp cần rõ ràng về quyền sử dụng, hãy liên hệ tác giả dự án.

## Liên hệ

Nếu bạn có câu hỏi, báo lỗi hoặc cần hỗ trợ, vui lòng mở issue trên repository hoặc liên hệ theo thông tin được cung cấp trong hồ sơ tác giả của repo.

---

*Cảm ơn bạn đã quan tâm và sử dụng Monica Discord Music Bot. Hãy sử dụng có trách nhiệm.*


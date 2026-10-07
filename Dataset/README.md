# Dataset và video

Dataset/archive local không được đưa lên Git. Inference trên video riêng
không cần tải dataset huấn luyện.

Chuẩn bị video `.mp4` trong thư mục tùy chọn và dùng:

```bash
cd Golf_Detection
python batch_golf_pipeline.py --input /path/to/videos --count 20 --device auto
python vlm/detect_club_videos.py --input /path/to/videos --count 2 --backend auto
```

Xem [hướng dẫn](../Golf_Detection/PORTABILITY.md) để cài đặt, chuẩn bị model
và tổ chức dữ liệu training GolfDB.

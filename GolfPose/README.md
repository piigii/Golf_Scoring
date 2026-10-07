# GolfPose checkpoints

Hai file trọng số local không được đưa lên Git:

- `golfpose_golfer_hrnetw48.pth`
- `golfpose_17+5_39.2_32.3_62.8.bin`

Pipeline mặc định đọc checkpoint tại `Golf_Detection/models/`.
Tải từ nguồn trong [PIPELINE.md](../Golf_Detection/PIPELINE.md), hoặc copy
trọng số local sang thư mục đó. Mã GolfPose được tải đúng revision bằng:

```bash
cd Golf_Detection
python scripts/fetch_golfpose.py
```

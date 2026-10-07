# GolfPose 2D → GolfPose 3D → SwingNet

Pipeline cho video đã crop, chứa một người và một cú swing. `test_video.mp4`
trong repository đáp ứng cách tổ chức đầu vào này. Pipeline dùng checkpoint
GolfPose chính thức; không thay bằng MediaPipe hoặc pose model chưa fine-tune.

## Cài đặt

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-pipeline.txt
python scripts/fetch_golfpose.py
```

`requirements-pipeline.lock.txt` ghi toàn bộ phiên bản đã cài khi chạy kiểm thử
trên macOS arm64/Python 3.11; có thể dùng file này thay file requirements ở trên
để tái lập môi trường.

Mã nguồn GolfPose và môi trường `.venv` được bỏ qua bởi Git. Phiên bản upstream
được ghi trong `third_party/GolfPose/.git`; nên giữ commit đó khi tái lập.
Checkout được dùng khi tích hợp: `0b749c6df4bb7f060ab42fa3dbd71de91bde17c1`.

Trên Apple Silicon, backend HRNet nạp trực tiếp backbone của MMPose cùng head
trong config GolfPose. Nó dùng MMCV-lite, không yêu cầu các detection op C++/CUDA
không liên quan. Affine crop, MSRA heatmap decoder và flip test theo config
GolfPose. Backend hiện hỗ trợ HRNet, không hỗ trợ ViTPose hoặc DEKR.

## Trọng số

Tạo thư mục `models/` và đặt các file:

| File | Nguồn chính thức |
|---|---|
| `golfpose_golfer_hrnetw48.pth` | [GolfPose 2D, 17 khớp người + 5 điểm gậy](http://gofile.me/4RvCV/YmRPLzEMc) |
| `golfpose_17+5_39.2_32.3_62.8.bin` | [GolfPose 3D, 22 keypoint](http://gofile.me/4RvCV/Xp00uwEBO) |
| `swingnet_1800.pth.tar` | [SwingNet](https://drive.google.com/file/d/1MBIDwHSM8OKRbxS8YfyRLnUBAdt0nupW/view) |

Các link lấy từ README của hai project. Hai checkpoint GolfPose đã được người
dùng cung cấp trong `models/` ngày 07/10/2026. Checkpoint 2D chứa HRNet với 22
keypoint; checkpoint 3D chứa MixSTE với 22 keypoint, embedding 512 và cửa sổ
243 frame. Trạng thái kiểm thử lưu tại `outputs/test_video/run_status.json`.
Checkpoint SwingNet cũng đã có trong workspace.

Không cần tải dataset GolfSwing để suy luận trên video riêng. Không cần
`mobilenet_v2.pth.tar` khi đã có checkpoint SwingNet đầy đủ.

## Chạy pipeline đầy đủ

```bash
.venv/bin/python golf_pipeline.py --video test_video.mp4 --device cpu
```

Đã chạy thành công pipeline đầy đủ trên CPU ngày 07/10/2026 với checkpoint
người dùng cung cấp: 264 frame, 29.970 FPS. Hai model nạp trọng số bằng
`strict=True`. Pose 2D có shape `(264, 22, 2)`, pose 3D có shape `(264, 22, 3)`;
toàn bộ tọa độ hữu hạn. Đã kiểm tra timestamp giữa hai tầng, pose tại 8 mốc
trong JSON và video output đủ 264 frame, kích thước 960×480.
Kết quả thật nằm trong `outputs/test_video/`; log tại `run.log`.

Các tùy chọn: `--device cpu|cuda|mps|auto`, `--output`,
`--pose2d-checkpoint`, `--pose3d-checkpoint`, `--pose-config`,
`--golfpose-dir`, `--seq-length` và `--threads`.
`auto` chọn CUDA → MPS (Apple Metal) → CPU. Xem [PORTABILITY.md](PORTABILITY.md) để cài theo nền tảng.

Nếu muốn chỉ dùng 17 khớp cơ thể, truyền config
`configs/mmpose/golfpose_person_hrnetw48.py` và các checkpoint 17 khớp
tương ứng trong README upstream. Không dùng lẫn thứ tự khớp COCO với GolfSwing.

## Xử lý 20 video đầu và ghép MP4

```bash
.venv/bin/python batch_golf_pipeline.py --input videos_160 --count 20
```

Chọn theo thứ tự số trong tên file (`0.mp4` đến `19.mp4`). Kết quả lưu trong
`outputs/first20/`: mỗi thư mục con chứa pose, JSON mốc và video riêng;
`golf_first20.mp4` ghép 20 video ở 1280×600, MPEG-4 (`mp4v`), 30 FPS. Dùng `--codec avc1` nếu OpenCV hỗ trợ H.264.
Video có annotation 2D trên hình gốc, skeleton 3D và bảng 8 mốc với frame,
thời gian, confidence. Bảng tô sáng mốc trong ±0,18 giây quanh dự đoán.
Các mốc gần nhau có thể được tô sáng đồng thời. Timeline hiển thị vị trí 8 mốc.

Nguồn có FPS hơi khác nhau được lấy mẫu theo thời gian về 30 FPS để ghép;
tọa độ và chỉ số frame trong NPZ/JSON riêng vẫn thuộc video gốc.
`batch_status.json` ghi danh sách nguồn, thời điểm bắt đầu mỗi clip trong MP4
tổng hợp, các mốc và kết quả kiểm tra. Khi chạy lại, script tái sử dụng video
đã hoàn tất, kiểm tra hash nguồn và dữ liệu rồi tạo lại MP4 tổng hợp.
Annotation là dự đoán của model, chưa phải nhãn ground truth do người gán.

Lần chạy ngày 07/10/2026 đã hoàn tất 20/20 video: MP4 H.264 có 4.347 frame,
30 FPS, thời lượng 144,9 giây. Đã decode kiểm tra toàn bộ video cuối và xác nhận
giữ đủ frame gốc cho 20 nguồn này sau khi lấy mẫu về 30 FPS. Cả 20 clip có 8 mốc
theo đúng thứ tự; 18/160 mốc có confidence dưới 0,3. Bảng tổng hợp 160 mốc nằm
ở `outputs/first20/events_summary.csv`, ảnh mẫu ở `preview_summary.jpg`.

## Kiểm tra SwingNet riêng

```bash
.venv/bin/python golf_pipeline.py --video test_video.mp4 --device cpu \
  --swingnet-only --output outputs/test_video_swingnet
```

Chế độ này chỉ xuất mốc thời gian và ảnh minh họa, không tạo kết quả pose.

Lần chạy CPU với video mẫu đã thành công: 264 frame, 29.970 FPS. Kết quả ở
`outputs/test_video_swingnet/events.json` và `events_montage.jpg`:

| Mốc | Frame (từ 0) | Giây | Confidence |
|---|---:|---:|---:|
| Address | 74 | 2.469 | 0.091 |
| Toe-up | 86 | 2.870 | 0.571 |
| Mid-backswing | 98 | 3.270 | 0.848 |
| Top | 114 | 3.804 | 0.707 |
| Mid-downswing | 132 | 4.404 | 0.867 |
| Impact | 143 | 4.771 | 0.974 |
| Mid-follow-through | 151 | 5.038 | 0.743 |
| Finish | 236 | 7.875 | 0.174 |

Các mốc theo đúng thứ tự thời gian. Address và Finish có confidence thấp, cần
kiểm tra thủ công; confidence không thay thế đánh giá bằng annotation chuẩn.

## Output đầy đủ

Trong `outputs/test_video/`:

- `pose2d.npz`: `keypoints` (T, 22, 2), `scores` (T, 22), tên khớp, timestamps.
- `pose3d.npz`: `keypoints` (T, 22, 3), tên khớp, timestamps, đơn vị và hệ tọa độ.
- `swingnet_probabilities.npy`: xác suất 9 lớp tại từng frame (T, 9).
- `events.json`: 8 mốc, confidence, frame, thời gian, pose tương ứng,
  thông tin video, SHA-256 của video và checkpoint.
- `annotated.mp4`: video ghép hình có skeleton 2D với khung xương 3D.
- `events_montage.jpg`, `event_*.jpg`: ảnh của 8 mốc.

Frame đánh số từ 0, giữ nguyên toàn bộ frame đã decode; không bỏ frame hay đổi
FPS đầu vào. JSON/NPZ ưu tiên timestamp từ decoder; nếu codec không cung cấp
timestamp tăng dần, dùng `frame_index / fps` và ghi rõ trong metadata.
Video minh họa được xuất ở FPS danh nghĩa, phù hợp với video CFR.

3D là dự đoán theo hệ tọa độ camera, đơn vị mét theo mô hình, đã trừ root
tại từng frame. Không có calibration camera cho video này, nên không phải phép
đo vị trí tuyệt đối trong không gian. Khớp ít tin cậy được ẩn trên overlay 2D;
mô hình 3D vẫn nhận toàn bộ keypoint dự đoán, không nhận confidence.

Đã xem ảnh tại Top và Impact của lần chạy thật. Phần gậy trong dự đoán 3D
cần đối chiếu thủ công; chưa có ground truth để xác nhận độ chính xác của
khung xương hoặc dùng trực tiếp làm tiêu chí chấm kỹ thuật.

SwingNet chọn argmax riêng cho từng sự kiện như code gốc. Pipeline giữ nguyên
dự đoán, ghi `events_strictly_ordered` để phát hiện mốc trùng/sai thứ tự.
Chưa có annotation chuẩn cho `test_video.mp4` để tính sai số hoặc PCE.

## Kiểm thử mã tích hợp

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Các fixture ngẫu nhiên trong unit test chỉ kiểm tra thiết bị, kích thước tensor,
ghép frame và xử lý chuỗi ngắn/dài; không được dùng làm kết quả pose của video.
Đã chạy 8 test và đều đạt, bao gồm backend HRNet trên CPU, adapter MixSTE,
ghép mốc theo timestamp và xuất video/ảnh bằng dữ liệu synthetic.

## Nguồn

- [GolfPose: From Regular Posture to Golf Swing Posture](https://github.com/MingHanLee/GolfPose)
- [GolfDB / SwingNet](https://github.com/wmcnally/golfdb)
- [MMPose](https://github.com/open-mmlab/mmpose)

Các thành phần và checkpoint giữ điều kiện sử dụng của tác giả gốc. Đọc LICENSE
của GolfPose trong checkout; giấy phép GolfDB được ghi trong README gốc.
